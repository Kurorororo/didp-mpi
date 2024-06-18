use didp_mpi::{
    AahParameters, AdditionalCommonParameters, DistributedFNode, HdHac, InitiationParameters,
    IsFloat, KeyValueStatistics, MpiAnytimeSearchParameters, Statistics,
};
use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{
    prelude::*,
    variable_type::{Numeric, OrderedContinuous},
};
use dypdl_heuristic_search::{
    search_algorithm::{util::TimeKeeper, SearchInput},
    Parameters,
};
use mpi::{environment::Universe, traits::*};
use std::hash::Hash;
use std::str::FromStr;
use std::{
    fmt::{Debug, Display},
    mem,
};
use yaml_rust::Yaml;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

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
    let (
        mut parameters,
        additional_parameters,
        initiation_parameters,
        aah_paramaters,
        ignore_initiation_nodes,
    ) = load_config_from_file::<T>(config_filename);

    if let Some(buffer_size) = additional_parameters.buffer_size {
        universe.set_buffer_size(buffer_size);
    }

    let f_evaluator_type = additional_parameters.f_evaluator_type;

    let (mut input, mut evaluators) = didp_mpi::make_input_and_mpi_dual_bound_evalautors(
        model,
        f_evaluator_type,
        parameters.primal_bound,
    );

    let initiator_input = SearchInput {
        node: input.node.clone().map(DistributedFNode::from),
        generator: input.generator.clone(),
        solution_suffix: input.solution_suffix,
    };

    let communicator = universe.world();

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

    let mut parameters = MpiAnytimeSearchParameters {
        controller_rank: 0,
        solution_filename,
        history_filename,
        count_bound_to_expanded: additional_parameters.count_bound_to_expanded,
        parameters,
    };

    let mut root_node = None;

    if !ignore_initiation_nodes {
        mem::swap(&mut input.node, &mut root_node);
    }

    let root_process = communicator.process_at_rank(0);

    if communicator.rank() == 0 {
        println!("Time for initialization: {}s", time_keeper.elapsed_time());
    }

    let search_time_keeper = TimeKeeper::default();

    if communicator.rank() == 0 {
        let initiation_result = didp_mpi::hac_initiator(
            initiator_input,
            &mut evaluators.local_successor_evaluator,
            &mut evaluators.base_cost_evaluator,
            initiation_parameters,
        );

        println!(
            "Initator generated {} nodes with {} seconds",
            initiation_result.n_nodes, initiation_result.solution.time
        );

        if initiation_result.solution.cost.is_some() {
            didp_mpi::write_solution(&initiation_result.solution, "solution.yaml");
        }

        if initiation_result.solution.is_optimal || initiation_result.solution.is_infeasible {
            if initiation_result.solution.is_infeasible {
                println!("Problem is infeasible");
            } else {
                println!("Optimal solution is found by the initiator");
                didp_mpi::dump_solution(&initiation_result.solution);
            }

            let mut finished_flag = true;
            root_process.broadcast_into(&mut finished_flag);

            return;
        }

        let mut finished_flag = false;
        root_process.broadcast_into(&mut finished_flag);

        if let Some(bound) = initiation_result.solution.best_bound {
            println!("Initial dual bound: {}", bound);
        }

        if let Some(time_limit) = parameters.parameters.time_limit {
            parameters.parameters.time_limit = Some(time_limit - search_time_keeper.elapsed_time());
        }

        let aah_result = didp_mpi::aah(
            &input.generator.model,
            &initiation_result,
            communicator.size(),
            &aah_paramaters,
        );

        if let Some(mut p) = aah_result.probability {
            println!("AAH p: {}", p);
            println!("Stddev: {}", aah_result.stddev);
            let mut p_flag = true;
            root_process.broadcast_into(&mut p_flag);
            root_process.broadcast_into(&mut p);
        } else {
            println!("AAH p: {}", 0.0);
            println!("Stddev: {}", aah_result.stddev);
            let mut p_flag = false;
            root_process.broadcast_into(&mut p_flag);
        }

        if let Some(table) = aah_result.table {
            let hash_function = didp_mpi::create_bytewise_zobrist_hash(table);
            let offset = search_time_keeper.elapsed_time();
            let mut solver =
                HdHac::new(input, evaluators, parameters, hash_function, &communicator);
            solver.set_time_offset(offset);

            if ignore_initiation_nodes {
                solver.initiate_without_distributing_nodes(initiation_result)
            } else {
                solver.distriute_initial_nodes(initiation_result, &aah_result.assignments);

                if let Some(node) = root_node {
                    solver.close_root_node(node);
                }
            }

            let (solution, statistics_list) = solver.search();

            if additional_parameters.count_bound_to_expanded {
                let bound_to_expanded = solver.gather_bound_to_expanded();

                KeyValueStatistics::dump_to_csv(&bound_to_expanded, "bound_to_expanded.csv")
                    .unwrap();
            }

            didp_mpi::dump_solution(&solution);

            println!(
                "Time to the final solution: {}s",
                time_keeper.elapsed_time()
            );

            didp_mpi::dump_statistics(&statistics_list);
            Statistics::dump_to_csv(&statistics_list, "statistics.csv").unwrap();
        } else {
            let hash_function = didp_mpi::create_fx_hash();
            let offset = search_time_keeper.elapsed_time();
            let mut solver =
                HdHac::new(input, evaluators, parameters, hash_function, &communicator);
            solver.set_time_offset(offset);

            if ignore_initiation_nodes {
                solver.initiate_without_distributing_nodes(initiation_result)
            } else {
                solver.distriute_initial_nodes(initiation_result, &aah_result.assignments);

                if let Some(node) = root_node {
                    solver.close_root_node(node);
                }
            }

            let (solution, statistics_list) = solver.search();

            if additional_parameters.count_bound_to_expanded {
                let bound_to_expanded = solver.gather_bound_to_expanded();

                KeyValueStatistics::dump_to_csv(&bound_to_expanded, "bound_to_expanded.csv")
                    .unwrap();
            }

            didp_mpi::dump_solution(&solution);

            println!(
                "Time to the final solution: {}s",
                time_keeper.elapsed_time()
            );

            didp_mpi::dump_statistics(&statistics_list);
            Statistics::dump_to_csv(&statistics_list, "statistics.csv").unwrap();
        };
    } else {
        let mut finished_flag = false;
        root_process.broadcast_into(&mut finished_flag);

        if finished_flag {
            return;
        }

        let mut p_flag = false;
        root_process.broadcast_into(&mut p_flag);

        if p_flag {
            let mut p = 0.0;
            root_process.broadcast_into(&mut p);
            let table = didp_mpi::create_abstract_bytewise_random_table(&input.generator.model, p);
            let hash_function = didp_mpi::create_bytewise_zobrist_hash(table);
            let offset = search_time_keeper.elapsed_time();
            let mut solver =
                HdHac::new(input, evaluators, parameters, hash_function, &communicator);
            solver.set_time_offset(offset);

            if !ignore_initiation_nodes {
                solver.receive_initial_nodes(0);

                if let Some(node) = root_node {
                    solver.close_root_node(node);
                }
            }

            solver.search();

            if additional_parameters.count_bound_to_expanded {
                solver.gather_bound_to_expanded();
            }
        } else {
            let hash_function = didp_mpi::create_fx_hash();
            let offset = search_time_keeper.elapsed_time();
            let mut solver =
                HdHac::new(input, evaluators, parameters, hash_function, &communicator);
            solver.set_time_offset(offset);

            if !ignore_initiation_nodes {
                solver.receive_initial_nodes(0);

                if let Some(node) = root_node {
                    solver.close_root_node(node);
                }
            }

            solver.search();

            if additional_parameters.count_bound_to_expanded {
                solver.gather_bound_to_expanded();
            }
        }
    }
}

fn load_config_from_file<T>(
    filename: &str,
) -> (
    Parameters<T>,
    AdditionalCommonParameters,
    InitiationParameters,
    AahParameters,
    bool,
)
where
    T: Numeric,
    <T as FromStr>::Err: Debug,
{
    let yaml = didp_mpi::read_config_yaml(filename);
    let map = yaml.as_hash().expect("Yaml file is not a hash");
    let parameters = didp_mpi::load_parameters_from_map::<T>(map);
    let additional_parameters = AdditionalCommonParameters::load_from_map(map);
    let key = Yaml::String(String::from("initiator"));
    let initiation_map = map.get(&key).expect("key 'initiator' is not found");
    let initiation_map = didp_yaml::util::get_map(initiation_map).expect("initiator is not a map");
    let initiation_parameters = InitiationParameters::load_from_map(initiation_map);
    let key = Yaml::String(String::from("aah"));
    let aah_map = map.get(&key).expect("key 'aah' is not found");
    let aah_map = didp_yaml::util::get_map(aah_map).expect("aah is not a map");
    let aah_parameters = AahParameters::load_from_map(aah_map);
    let ignore_initiation_nodes =
        didp_mpi::load_bool_from_map(map, "ignore_initiation_nodes").unwrap_or(false);

    (
        parameters,
        additional_parameters,
        initiation_parameters,
        aah_parameters,
        ignore_initiation_nodes,
    )
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
