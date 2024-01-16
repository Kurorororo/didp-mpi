use didp_mpi::{
    dump_solution, load_parameters_from_map, read_model, AdditionalCommonParameters,
    DistributedFNode, DistributedFNodeMessage, HashType, HdAcps, IsFloat,
    MpiAnytimeSearchParameters,
};
use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::prelude::*;
use dypdl::variable_type::{Numeric, OrderedContinuous};
use dypdl_heuristic_search::search_algorithm::data_structure::HashableSignatureVariables;
use dypdl_heuristic_search::search_algorithm::{SearchInput, SuccessorGenerator, TransitionWithId};
use dypdl_heuristic_search::{FEvaluatorType, Parameters, ProgressiveSearchParameters};
use mpi::environment::Universe;
use mpi::traits::*;
use std::fmt::{Debug, Display};
use std::fs;
use std::rc::Rc;
use std::str::FromStr;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main_with_cost_type_and_hash_function<T, H>(
    universe: Universe,
    model: Model,
    mut parameters: Parameters<T>,
    progressive_parameters: ProgressiveSearchParameters,
    f_evaluator_type: FEvaluatorType,
    hash_function: H,
) where
    T: Numeric + IsFloat + Ord + Display,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    H: Fn(&HashableSignatureVariables) -> u64,
{
    let model = Rc::new(model);
    let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
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
    let node = DistributedFNodeMessage::generate_root_node(
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
    let transition_evaluator = move |node: &DistributedFNode<_>, transition: &_, primal_bound| {
        node.generate_sendable_successor_node(
            transition,
            &model,
            &h_evaluator,
            &f_evaluator,
            primal_bound,
        )
    };

    let communicator = universe.world();

    let solution_filename = if communicator.rank() == 0 {
        Some(String::from("solution.yaml"))
    } else {
        None
    };

    parameters.quiet |= communicator.rank() != 0;
    let parameters = MpiAnytimeSearchParameters {
        controller_rank: 0,
        solution_filename,
        parameters,
    };

    let solver = HdAcps::new(
        input,
        transition_evaluator,
        base_cost_evaluator,
        parameters,
        progressive_parameters,
        hash_function,
        &communicator,
    );
    let (solution, statistics_list) = solver.search();

    if communicator.rank() == 0 {
        dump_solution(&solution);
        let statistics_yaml = serde_yaml::to_string(&statistics_list).unwrap();
        fs::write("statistics.yaml", statistics_yaml).unwrap();
    }
}

fn main_with_cost_type<T>(mut universe: Universe, model: Model, config_filename: &str)
where
    T: Numeric + IsFloat + Ord + Display,
    CostToDump: From<T>,
    <T as FromStr>::Err: Debug,
{
    let (parameters, progressive_search_parameters, additional_parameters) =
        load_config_from_file::<T>(config_filename);

    if let Some(buffer_size) = additional_parameters.buffer_size {
        universe.set_buffer_size(buffer_size);
    }

    let f_evaluator_type = additional_parameters.f_evaluator_type;

    match additional_parameters.hash_type {
        HashType::Fx => {
            let hash_function = didp_mpi::create_fx_hash();
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                progressive_search_parameters,
                f_evaluator_type,
                hash_function,
            );
        }
        HashType::SetZobrist => {
            let hash_function = didp_mpi::create_set_zobrist_hash(
                &model,
                additional_parameters.zobrist_zero_probability,
            );
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                progressive_search_parameters,
                f_evaluator_type,
                hash_function,
            );
        }
        HashType::SetZobristWithOthers => {
            let hash_function = didp_mpi::create_set_zobrist_hash_with_others(
                &model,
                additional_parameters.zobrist_zero_probability,
            );
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                progressive_search_parameters,
                f_evaluator_type,
                hash_function,
            );
        }
    }
}

fn load_config_from_file<T>(
    filename: &str,
) -> (
    Parameters<T>,
    ProgressiveSearchParameters,
    AdditionalCommonParameters,
)
where
    T: Numeric,
    <T as FromStr>::Err: Debug,
{
    let config = fs::read_to_string(filename).unwrap_or_else(|e| {
        panic!("Couldn't read a config file: {:?}", e);
    });
    let config = yaml_rust::YamlLoader::load_from_str(&config).unwrap_or_else(|e| {
        panic!("Config file must be in YAML format: {:?}", e);
    });
    assert_eq!(config.len(), 1);
    let yaml = &config[0];
    let map = yaml.as_hash().expect("Yaml file is not a hash");
    let parameters = load_parameters_from_map::<T>(map);
    let additional_common_parameters = AdditionalCommonParameters::load_from_map(map);

    let init = match map.get(&yaml_rust::Yaml::from_str("init")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            panic!("expected Integer for `init`, but found `{:?}`", value)
        }
        None => 1,
    };
    let step = match map.get(&yaml_rust::Yaml::from_str("step")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            panic!("expected Integer for `step`, but found `{:?}`", value)
        }
        None => 1,
    };
    let bound = match map.get(&yaml_rust::Yaml::from_str("width_bound")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        Some(value) => {
            panic!(
                "expected Integer for `width_bound`, but found `{:?}`",
                value
            )
        }
        None => None,
    };
    let reset = match map.get(&yaml_rust::Yaml::from_str("reset")) {
        Some(yaml_rust::Yaml::Boolean(value)) => !(*value),
        None => false,
        value => {
            panic!("expected Boolean for `reset`, but found `{:?}`", value)
        }
    };
    let progressive_parameters = ProgressiveSearchParameters {
        init,
        step,
        bound,
        reset,
    };

    (
        parameters,
        progressive_parameters,
        additional_common_parameters,
    )
}

fn main() {
    let universe = mpi::initialize().unwrap();

    let mut args = std::env::args();
    args.next();
    let model = read_model(&mut args);
    let config_filename = args.next().expect("Config filename is not specified");

    match model.cost_type {
        CostType::Integer => main_with_cost_type::<Integer>(universe, model, &config_filename),
        CostType::Continuous => {
            main_with_cost_type::<OrderedContinuous>(universe, model, &config_filename)
        }
    }
}
