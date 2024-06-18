use didp_mpi::{
    AdditionalCommonParameters, HashType, HdHac, IsFloat, KeyValueStatistics,
    MpiAnytimeSearchParameters, NodeDatatype, Statistics,
};
use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{
    prelude::*,
    variable_type::{Numeric, OrderedContinuous},
};
use dypdl_heuristic_search::{
    search_algorithm::{data_structure::HashableSignatureVariables, util::TimeKeeper},
    FEvaluatorType, Parameters,
};
use mpi::{environment::Universe, traits::*};
use std::fmt::{Debug, Display};
use std::hash::Hash;
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
    f_evaluator_type: FEvaluatorType,
    hash_function: H,
    count_bound_to_expanded: bool,
    time_keeper: &TimeKeeper,
) where
    T: Numeric + IsFloat + Ord + Display + Hash,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    H: Fn(&HashableSignatureVariables) -> u64,
{
    let (input, evaluators) = didp_mpi::make_input_and_mpi_dual_bound_evalautors(
        model,
        f_evaluator_type,
        parameters.primal_bound,
    );

    let communicator = universe.world();

    if communicator.rank() == 0 && !parameters.quiet {
        if let Some(node) = &input.node {
            println!(
                "Initial dual bound: {}",
                node.bound(&input.generator.model).unwrap()
            );
        }
    }

    let solution_filename = if communicator.rank() == 0 {
        Some(String::from("solution.yaml"))
    } else {
        None
    };
    let history_filename = if communicator.rank() == 0 {
        Some(String::from("history.csv"))
    } else {
        None
    };

    parameters.quiet |= communicator.rank() != 0;
    let parameters = MpiAnytimeSearchParameters {
        controller_rank: 0,
        solution_filename,
        history_filename,
        count_bound_to_expanded,
        parameters,
    };

    let mut solver = HdHac::new(input, evaluators, parameters, hash_function, &communicator);
    let (solution, statistics_list) = solver.search();

    if communicator.rank() == 0 {
        didp_mpi::dump_solution(&solution);

        println!(
            "Time to the final solution: {}s",
            time_keeper.elapsed_time()
        );

        didp_mpi::dump_statistics(&statistics_list);
        Statistics::dump_to_csv(&statistics_list, "statistics.csv").unwrap();
    }

    if count_bound_to_expanded {
        let bound_to_expanded = solver.gather_bound_to_expanded();

        if communicator.rank() == 0 {
            KeyValueStatistics::dump_to_csv(&bound_to_expanded, "bound_to_expanded.csv").unwrap();
        }
    }
}

fn main_with_cost_type<T>(
    mut universe: Universe,
    model: Model,
    config_filename: &str,
    time_keeper: &TimeKeeper,
) where
    T: Numeric + IsFloat + Ord + Display + Hash,
    CostToDump: From<T>,
    <T as FromStr>::Err: Debug,
{
    let yaml = didp_mpi::read_config_yaml(config_filename);
    let map = yaml.as_hash().expect("Yaml file is not a hash");
    let parameters = didp_mpi::load_parameters_from_map::<T>(map);
    let additional_parameters = AdditionalCommonParameters::load_from_map(map);

    if let Some(buffer_size) = additional_parameters.buffer_size {
        universe.set_buffer_size(buffer_size);
    }

    let f_evaluator_type = additional_parameters.f_evaluator_type;

    match additional_parameters.hash_type {
        HashType::Wy => {
            let hash_function = didp_mpi::create_wyhash();
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                f_evaluator_type,
                hash_function,
                additional_parameters.count_bound_to_expanded,
                time_keeper,
            );
        }
        HashType::MaskedWy => {
            let masks = didp_mpi::create_set_masks(
                &model,
                additional_parameters.abstraction_probability.unwrap_or(0.0),
            );
            let hash_function = didp_mpi::create_masked_wyhash(masks);
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                f_evaluator_type,
                hash_function,
                additional_parameters.count_bound_to_expanded,
                time_keeper,
            );
        }
        HashType::Fx => {
            let hash_function = didp_mpi::create_fx_hash();
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                f_evaluator_type,
                hash_function,
                additional_parameters.count_bound_to_expanded,
                time_keeper,
            );
        }
        HashType::MaskedFx => {
            let masks = didp_mpi::create_set_masks(
                &model,
                additional_parameters.abstraction_probability.unwrap_or(0.0),
            );
            let hash_function = didp_mpi::create_masked_fx_hash(masks);
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                f_evaluator_type,
                hash_function,
                additional_parameters.count_bound_to_expanded,
                time_keeper,
            );
        }
        HashType::SetZobrist => {
            let table = didp_mpi::create_abstract_bytewise_random_table(
                &model,
                additional_parameters.abstraction_probability.unwrap_or(0.0),
            );
            let hash_function = didp_mpi::create_bytewise_zobrist_hash(table);
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                f_evaluator_type,
                hash_function,
                additional_parameters.count_bound_to_expanded,
                time_keeper,
            );
        }
        HashType::SetZobristWithOthers => {
            let table = didp_mpi::create_abstract_bytewise_random_table(
                &model,
                additional_parameters.abstraction_probability.unwrap_or(0.0),
            );
            let hash_function = didp_mpi::create_bytewise_zobrist_hash_with_others(table);
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                f_evaluator_type,
                hash_function,
                additional_parameters.count_bound_to_expanded,
                time_keeper,
            );
        }
        HashType::ThreeBitsFieldZobrist => {
            let hash_function = didp_mpi::create_3bits_field_zobrist_hash(
                &model,
                additional_parameters.abstraction_probability.unwrap_or(0.0),
            );
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                f_evaluator_type,
                hash_function,
                additional_parameters.count_bound_to_expanded,
                time_keeper,
            );
        }
        HashType::FourBitsFieldZobrist => {
            let hash_function = didp_mpi::create_4bits_field_zobrist_hash(
                &model,
                additional_parameters.abstraction_probability.unwrap_or(0.0),
            );
            main_with_cost_type_and_hash_function(
                universe,
                model,
                parameters,
                f_evaluator_type,
                hash_function,
                additional_parameters.count_bound_to_expanded,
                time_keeper,
            );
        }
    }
}

fn main() {
    let time_keeper = TimeKeeper::default();
    let universe = mpi::initialize().unwrap();
    let rank = universe.world().rank();

    if rank == 0 {
        println!("Time to initialize MPI: {}s", time_keeper.elapsed_time());
    }

    let mut args = std::env::args();
    args.next();
    let model = didp_mpi::read_model(&mut args);
    let config_filename = args.next().expect("Config filename is not specified");

    match model.cost_type {
        CostType::Integer => {
            main_with_cost_type::<Integer>(universe, model, &config_filename, &time_keeper)
        }
        CostType::Continuous => main_with_cost_type::<OrderedContinuous>(
            universe,
            model,
            &config_filename,
            &time_keeper,
        ),
    }

    if rank == 0 {
        println!("Total time: {}s", time_keeper.elapsed_time());
    }
}
