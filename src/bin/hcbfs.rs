use didp_mpi::Hcbfs;
use didp_yaml::heuristic_search_solver::{self, CostToDump};
use dypdl::{
    prelude::*,
    variable_type::{Numeric, OrderedContinuous},
};
use dypdl_heuristic_search::{
    search_algorithm::{FNode, SearchInput, SuccessorGenerator},
    FEvaluatorType, Parameters, Search,
};
use std::fmt::{Debug, Display};
use std::fs;
use std::rc::Rc;
use std::str::FromStr;
use yaml_rust::{Yaml, YamlLoader};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main_with_cost_type<T>(model: Model, config_filename: &str)
where
    T: Numeric + Ord + Display + 'static,
    CostToDump: From<T>,
    <T as FromStr>::Err: Debug,
{
    let (parameters, f_evaluator_type) = load_config_from_file::<T>(config_filename);

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

    let solver: Box<dyn Search<T>> = Box::new(Hcbfs::new(
        input,
        transition_evaluator,
        base_cost_evaluator,
        parameters,
    ));

    let solution =
        heuristic_search_solver::solve_and_dump_solutions(solver, "history.csv", "solution.csv")
            .unwrap();

    didp_mpi::dump_solution(&solution);
}

fn load_config_from_file<T>(filename: &str) -> (Parameters<T>, FEvaluatorType)
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

    (parameters, f_evaluator_type)
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
