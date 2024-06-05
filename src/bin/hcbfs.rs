use didp_mpi::{Hcbfs, IsFloat, KeyValueStatistics};
use didp_yaml::heuristic_search_solver::{CostToDump, SolutionToDump};
use dypdl::{
    prelude::*,
    variable_type::{Numeric, OrderedContinuous},
};
use dypdl_heuristic_search::{
    search_algorithm::{FNode, SearchInput, SuccessorGenerator},
    FEvaluatorType, Parameters, Search,
};
use std::fs;
use std::hash::Hash;
use std::rc::Rc;
use std::str::FromStr;
use std::{error::Error, io::Write};
use std::{
    fmt::{Debug, Display},
    fs::OpenOptions,
};
use yaml_rust::{Yaml, YamlLoader};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn solve_and_dump_solutions<T, S: Search<T>>(
    solver: &mut S,
    history_filename: &str,
    solution_filename: &str,
) -> Result<dypdl_heuristic_search::Solution<T>, Box<dyn Error>>
where
    T: Numeric + Ord + Display + 'static,
    <T as FromStr>::Err: Debug,
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

fn main_with_cost_type<T>(model: Model, config_filename: &str)
where
    T: Numeric + Ord + Display + Hash + IsFloat + 'static,
    CostToDump: From<T>,
    <T as FromStr>::Err: Debug,
{
    let (parameters, f_evaluator_type, count_bound_to_expanded) =
        load_config_from_file::<T>(config_filename);

    let model = Rc::new(model);
    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost: T, base_cost| f_evaluator_type.eval(cost, base_cost);
    let cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };
    let h_model = model.clone();
    let h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
    let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
    let node = FNode::<_>::generate_root_node(
        model.target.clone(),
        cost,
        &model,
        &h_evaluator,
        &f_evaluator,
        parameters.primal_bound,
    );
    let input = SearchInput {
        node,
        generator,
        solution_suffix: &[],
    };
    let transition_evaluator =
        move |node: &FNode<_>, transition, registry: &mut _, primal_bound| {
            node.insert_successor_node(
                transition,
                registry,
                &h_evaluator,
                &f_evaluator,
                primal_bound,
            )
        };

    let mut solver = Hcbfs::new(
        input,
        transition_evaluator,
        base_cost_evaluator,
        parameters,
        count_bound_to_expanded,
    );

    let solution = solve_and_dump_solutions(&mut solver, "history.csv", "solution.yaml").unwrap();

    didp_mpi::dump_solution(&solution);

    if count_bound_to_expanded {
        let bound_to_expanded = KeyValueStatistics::from(solver.get_bound_to_expanded());
        KeyValueStatistics::dump_to_csv(&[bound_to_expanded], "bound_to_expanded.csv").unwrap();
    }
}

fn load_config_from_file<T>(filename: &str) -> (Parameters<T>, FEvaluatorType, bool)
where
    T: Numeric,
    <T as FromStr>::Err: Debug,
{
    let config = fs::read_to_string(filename).unwrap_or_else(|e| {
        panic!("Couldn't read a config file: {:?}", e);
    });
    let config = YamlLoader::load_from_str(&config).unwrap_or_else(|e| {
        panic!("Config file must be in YAML format: {:?}", e);
    });
    assert_eq!(config.len(), 1);
    let yaml = &config[0];
    let map = yaml.as_hash().expect("Yaml file is not a hash");
    let parameters = didp_mpi::load_parameters_from_map::<T>(map);

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

    let count_bound_to_expanded = map
        .get(&Yaml::String("count_bound_to_expanded".into()))
        .map(|t| {
            t.as_bool()
                .expect("count_bound_to_expanded must be boolean")
        })
        .unwrap_or(false);

    (parameters, f_evaluator_type, count_bound_to_expanded)
}

fn main() {
    let mut args = std::env::args();
    args.next();
    let model = didp_mpi::read_model(&mut args);
    let config_filename = args.next().expect("Config filename is not specified");

    match model.cost_type {
        CostType::Integer => main_with_cost_type::<Integer>(model, &config_filename),
        CostType::Continuous => main_with_cost_type::<OrderedContinuous>(model, &config_filename),
    }
}
