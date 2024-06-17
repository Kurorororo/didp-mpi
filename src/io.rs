use didp_yaml::{
    dypdl_parser,
    heuristic_search_solver::{CostToDump, SolutionToDump},
};
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, ProgressiveSearchParameters,
    Search, Solution,
};
use linked_hash_map::LinkedHashMap;
use std::fs;
use std::process;
use std::str::FromStr;
use std::{env::Args, error::Error, fs::OpenOptions};
use std::{
    fmt::{Debug, Display},
    io::Write,
};
use yaml_rust::{Yaml, YamlLoader};

use crate::{aah::AahParameters, InitiationParameters, Statistics};

pub fn read_model(args: &mut Args) -> Model {
    let domain = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a domain file name.");
        process::exit(1);
    });
    let problem = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a problem file name.");
        process::exit(1);
    });

    let domain = fs::read_to_string(domain).unwrap_or_else(|e| {
        eprintln!("Couldn't read a domain file: {:?}", e);
        process::exit(1);
    });
    let domain = YamlLoader::load_from_str(&domain).unwrap_or_else(|e| {
        eprintln!("Couldn't read a domain file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(domain.len(), 1);
    let domain = &domain[0];

    let problem = fs::read_to_string(problem).unwrap_or_else(|e| {
        eprintln!("Could'nt read a problem file: {:?}", e);
        process::exit(1);
    });
    let problem = YamlLoader::load_from_str(&problem).unwrap_or_else(|e| {
        eprintln!("Couldn't read a problem file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(problem.len(), 1);
    let problem = &problem[0];

    dypdl_parser::load_model_from_yaml(domain, problem).unwrap_or_else(|e| {
        eprintln!("Couldn't load a model: {:?}", e);
        process::exit(1);
    })
}

pub fn read_config_yaml(filename: &str) -> Yaml {
    let config = fs::read_to_string(filename).unwrap_or_else(|e| {
        panic!("Couldn't read a config file: {:?}", e);
    });
    let mut config = yaml_rust::YamlLoader::load_from_str(&config).unwrap_or_else(|e| {
        panic!("Config file must be in YAML format: {:?}", e);
    });
    assert_eq!(config.len(), 1);
    config.remove(0)
}

pub fn solve_and_dump_solutions<T, S>(
    solver: &mut S,
    history_filename: &str,
    solution_filename: &str,
) -> Result<Solution<T>, Box<dyn Error>>
where
    T: Numeric + Ord + Display + 'static,
    <T as FromStr>::Err: Debug,
    S: Search<T>,
    CostToDump: From<T>,
{
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(history_filename)?;

    loop {
        let (solution, terminated) = solver.search_next()?;

        if let Some(cost) = solution.cost {
            let line = format!(
                "{}, {}, {}, {}\n",
                solution.time, cost, solution.expanded, solution.generated
            );
            file.write_all(line.as_bytes())?;
            let solution_to_dump = SolutionToDump::from(solution.clone());
            solution_to_dump.dump_to_file(solution_filename)?;
        }

        if terminated {
            return Ok(solution);
        }
    }
}

pub fn write_solution<T, V>(solution: &Solution<T, V>, filename: &str)
where
    T: Numeric,
    CostToDump: From<T>,
    V: Clone,
    Transition: From<V>,
{
    let solution = Solution {
        cost: solution.cost,
        transitions: solution
            .transitions
            .iter()
            .map(|t| Transition::from(t.clone()))
            .collect(),
        is_optimal: solution.is_optimal,
        is_infeasible: solution.is_infeasible,
        best_bound: solution.best_bound,
        time_out: solution.time_out,
        expanded: solution.expanded,
        generated: solution.generated,
        time: solution.time,
    };
    let solution_to_dump = SolutionToDump::from(solution);
    solution_to_dump.dump_to_file(filename).unwrap();
}

pub fn dump_solution<T, V>(solution: &Solution<T, V>)
where
    T: Numeric + Display,
    V: Clone,
    Transition: From<V>,
{
    let expanded = solution.expanded;
    let generated = solution.generated;
    let search_time = solution.time;

    if let Some(cost) = solution.cost {
        println!("transitions:");
        for transition in &solution.transitions {
            let transition = Transition::from(transition.clone());
            println!("{}", transition.get_full_name());
        }

        println!("cost: {}", cost);
        if solution.is_optimal {
            println!("optimal cost: {}", cost);
        } else if let Some(bound) = solution.best_bound {
            println!("best bound: {}", bound);
        }
    } else if solution.is_infeasible {
        println!("The problem is infeasible.");
    } else {
        println!("Could not find a solution.");

        if let Some(bound) = solution.best_bound {
            println!("best bound: {}", bound);
        }
    }

    println!("Expanded: {}", expanded);
    println!("Generated: {}", generated);
    println!("Search time: {}s", search_time);
}

pub fn dump_statistics(statistics_list: &[Statistics]) {
    let kept: usize = statistics_list.iter().map(|s| s.kept).sum();
    let sent: usize = statistics_list.iter().map(|s| s.sent).sum();
    let dominated_before_closed: usize = statistics_list
        .iter()
        .map(|s| s.dominated_before_closed)
        .sum();
    let dominated_after_closed: usize = statistics_list
        .iter()
        .map(|s| s.dominated_after_closed)
        .sum();
    let max_expanded = statistics_list.iter().map(|s| s.expanded).max();
    let min_expanded = statistics_list.iter().map(|s| s.expanded).min();
    let max_received = statistics_list.iter().map(|s| s.received).max();
    let min_received = statistics_list.iter().map(|s| s.received).min();

    println!("Kept: {}", kept);
    println!("Sent: {}", sent);
    println!("Dominated before closed: {}", dominated_before_closed);
    println!("Dominated after closed: {}", dominated_after_closed);

    if let Some(max_expanded) = max_expanded {
        println!("Max expanded: {}", max_expanded);
    }

    if let Some(min_expanded) = min_expanded {
        println!("Min expanded: {}", min_expanded);
    }

    if let Some(max_received) = max_received {
        println!("Max received: {}", max_received);
    }

    if let Some(min_received) = min_received {
        println!("Min received: {}", min_received);
    }
}

pub fn load_bool_from_map(map: &LinkedHashMap<Yaml, Yaml>, key: &str) -> Option<bool> {
    map.get(&Yaml::from_str(key)).map(|t| {
        t.as_bool()
            .unwrap_or_else(|| panic!("{} must be boolean", key))
    })
}

pub fn load_usize_from_map(map: &LinkedHashMap<Yaml, Yaml>, key: &str) -> Option<usize> {
    map.get(&Yaml::from_str(key)).map(|t| {
        t.as_i64()
            .map_or_else(|| panic!("{} must be integer", key), |i| i as usize)
    })
}

pub fn load_f64_from_map(map: &LinkedHashMap<Yaml, Yaml>, key: &str) -> Option<f64> {
    map.get(&Yaml::from_str(key)).map(|t| {
        t.as_f64()
            .unwrap_or_else(|| panic!("{} must be float", key))
    })
}

pub fn load_parameters_from_map<T>(map: &LinkedHashMap<Yaml, Yaml>) -> Parameters<T>
where
    T: Numeric,
    <T as FromStr>::Err: Debug,
{
    let primal_bound = match map.get(&yaml_rust::Yaml::from_str("primal_bound")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(T::from_integer(*value as Integer)),
        Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().unwrap()),
        None => None,
        value => {
            panic!("expected Integer or Real, but found `{:?}`", value)
        }
    };
    let time_limit = map
        .get(&yaml_rust::Yaml::from_str("time_limit"))
        .map(|value| didp_yaml::util::get_numeric(value).unwrap());
    let quiet = load_bool_from_map(map, "quiet").unwrap_or(false);
    let get_all_solutions = load_bool_from_map(map, "get_all_solutions").unwrap_or(false);
    let initial_registry_capacity =
        load_usize_from_map(map, "initial_registry_capacity").or(Some(1000000));

    Parameters {
        primal_bound,
        time_limit,
        get_all_solutions,
        quiet,
        initial_registry_capacity,
    }
}

pub fn load_f_evaluator_type_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> FEvaluatorType {
    map.get(&Yaml::String("f_evaluator_type".into()))
        .map(|t| {
            let t = t.as_str().expect("f_evaluator_type must be string");
            match t {
                "+" => FEvaluatorType::Plus,
                "*" => FEvaluatorType::Product,
                "max" => FEvaluatorType::Max,
                "min" => FEvaluatorType::Min,
                _ => panic!("Invalid f_evaluator_type {:?}", t),
            }
        })
        .unwrap_or(FEvaluatorType::Plus)
}

pub fn load_cabs_parameters_from_map<T>(map: &LinkedHashMap<Yaml, Yaml>) -> CabsParameters<T>
where
    T: Numeric,
    <T as FromStr>::Err: Debug,
{
    let parameters = load_parameters_from_map::<T>(map);
    let beam_size = load_usize_from_map(map, "initial_beam_size").unwrap_or(1);
    let keep_all_layers = load_bool_from_map(map, "keep_all_layers").unwrap_or(false);
    let max_beam_size = load_usize_from_map(map, "max_beam_size");
    let beam_search_parameters = BeamSearchParameters {
        parameters,
        beam_size,
        keep_all_layers,
    };

    CabsParameters {
        max_beam_size,
        beam_search_parameters,
    }
}

pub fn load_progressive_parameters_from_map(
    map: &LinkedHashMap<Yaml, Yaml>,
) -> ProgressiveSearchParameters {
    let init = load_usize_from_map(map, "init").unwrap_or(1);
    let step = load_usize_from_map(map, "step").unwrap_or(1);
    let bound = load_usize_from_map(map, "width_bound");
    let reset = load_bool_from_map(map, "reset").unwrap_or(false);

    ProgressiveSearchParameters {
        init,
        step,
        bound,
        reset,
    }
}

pub enum HashType {
    Fx,
    MaskedFx,
    SetZobrist,
    SetZobristWithOthers,
    ThreeBitsFieldZobrist,
    FourBitsFieldZobrist,
}

pub struct AdditionalCommonParameters {
    pub f_evaluator_type: FEvaluatorType,
    pub buffer_size: Option<usize>,
    pub hash_type: HashType,
    pub abstraction_probability: Option<f64>,
    pub count_bound_to_expanded: bool,
}

impl AdditionalCommonParameters {
    pub fn load_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> Self {
        let f_evaluator_type = load_f_evaluator_type_from_map(map);
        let buffer_size = load_usize_from_map(map, "buffer_size");

        let hash_type = map
            .get(&Yaml::String("hash_type".into()))
            .map(|t| {
                let t = t.as_str().expect("hash_type must be string");
                match t {
                    "fx" => HashType::Fx,
                    "masked_fx" => HashType::MaskedFx,
                    "set_zobrist" => HashType::SetZobrist,
                    "set_zobrist_with_others" => HashType::SetZobristWithOthers,
                    "4bits_field_zobrist" => HashType::FourBitsFieldZobrist,
                    "3bits_field_zobrist" => HashType::ThreeBitsFieldZobrist,
                    _ => panic!("Invalid hash_type {:?}", t),
                }
            })
            .unwrap_or(HashType::Fx);

        let abstraction_probability = load_f64_from_map(map, "abstraction_probability")
            .or_else(|| load_f64_from_map(map, "zobrist_zero_probability"));
        let count_bound_to_expanded =
            load_bool_from_map(map, "count_bound_to_expanded").unwrap_or(false);

        Self {
            f_evaluator_type,
            buffer_size,
            hash_type,
            abstraction_probability,
            count_bound_to_expanded,
        }
    }
}

impl InitiationParameters {
    pub fn load_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> Self {
        let time_limit = map
            .get(&yaml_rust::Yaml::from_str("time_limit"))
            .map(|value| didp_yaml::util::get_numeric(value).unwrap());
        let node_limit = load_usize_from_map(map, "node_limit");
        let quiet = load_bool_from_map(map, "quiet").unwrap_or(false);

        Self {
            time_limit,
            node_limit,
            quiet,
        }
    }
}

impl AahParameters {
    pub fn load_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> Self {
        let max_probability = load_f64_from_map(map, "max_probability").unwrap_or(1.0);
        let step_size = load_f64_from_map(map, "step_size").unwrap_or(0.1);
        let threshold_ratio_to_average = load_f64_from_map(map, "threshold_ratio_to_average");
        let threshold_ratio_to_base = load_f64_from_map(map, "threshold_ratio_to_base");

        Self {
            max_probability,
            step_size,
            threshold_ratio_to_average,
            threshold_ratio_to_base,
        }
    }
}
