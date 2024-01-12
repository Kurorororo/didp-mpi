use didp_yaml::heuristic_search_solver::{CostToDump, SolutionToDump};
use didp_yaml::{dypdl_parser, util::get_numeric};
use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{FEvaluatorType, Parameters, Solution};
use linked_hash_map::LinkedHashMap;
use std::env::Args;
use std::fmt::{Debug, Display};
use std::fs;
use std::process;
use std::str::FromStr;
use yaml_rust::{Yaml, YamlLoader};

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

pub fn load_parameters_from_map<T: Numeric>(map: &LinkedHashMap<Yaml, Yaml>) -> Parameters<T>
where
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
        .map(|value| get_numeric(value).unwrap());
    let quiet = match map.get(&yaml_rust::Yaml::from_str("quiet")) {
        Some(Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            panic!("expected Boolean, but found `{:?}`", value)
        }
    };
    let get_all_solutions = match map.get(&yaml_rust::Yaml::from_str("get_all_solutions")) {
        Some(Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            panic!("expected Boolean, but found `{:?}`", value)
        }
    };
    let initial_registry_capacity = match map.get(&Yaml::from_str("initial_registry_capacity")) {
        Some(Yaml::Integer(value)) => Some(*value as usize),
        None => Some(1000000),
        value => {
            panic!(
                "expected Integer for `initial_registry_capacity`, but found `{:?}`",
                value
            )
        }
    };
    Parameters {
        primal_bound,
        time_limit,
        get_all_solutions,
        quiet,
        initial_registry_capacity,
    }
}
pub enum HashType {
    Fx,
    SetZobrist,
    SetZobristWithOthers,
}

pub struct AdditionalCommonParameters {
    pub f_evaluator_type: FEvaluatorType,
    pub buffer_size: Option<usize>,
    pub hash_type: HashType,
    pub zobrist_zero_probability: Option<f64>,
}

impl AdditionalCommonParameters {
    pub fn load_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> Self {
        let f_evaluator_type = map
            .get(&Yaml::String("f_evaluator_type".into()))
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
            .unwrap_or(FEvaluatorType::Plus);

        let buffer_size = map
            .get(&Yaml::String("buffer_size".into()))
            .map(|x| x.as_i64().expect("buffer_size must be an integer") as usize);

        let hash_type = map
            .get(&Yaml::String("hash_type".into()))
            .map(|t| {
                let t = t.as_str().expect("hash_type must be string");
                match t {
                    "fx" => HashType::Fx,
                    "set_zobrist" => HashType::SetZobrist,
                    "set_zobrist_with_others" => HashType::SetZobristWithOthers,
                    _ => panic!("Invalid hash_type {:?}", t),
                }
            })
            .unwrap_or(HashType::Fx);

        let zobrist_zero_probability = map
            .get(&Yaml::String("zobrist_zero_probability".into()))
            .map(|x| {
                x.as_f64()
                    .expect("zobrist_zero_probability must be a float")
            });

        Self {
            f_evaluator_type,
            buffer_size,
            hash_type,
            zobrist_zero_probability,
        }
    }
}
