use didp_mpi::{
    AdditionalCommonParameters, HashType, HdHac, HdHacMemoryStatistics, IsFloat,
    MpiAnytimeSearchParameters, NodeMessage, Statistics, TAG_EXPANSION_STATISTICS,
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
use linked_hash_map::LinkedHashMap;
use mpi::{environment::Universe, traits::*, Rank};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::str::FromStr;
use yaml_rust::Yaml;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone, Debug)]
enum MemoryMonitoringProcesses {
    All,
    Ranks(Vec<Rank>),
}

#[derive(Clone, Debug)]
struct MemoryMonitoringParameters {
    enabled: bool,
    processes: MemoryMonitoringProcesses,
    interval: f64,
}

#[derive(Clone, Debug)]
struct InstrumentationParameters {
    memory_monitoring: MemoryMonitoringParameters,
    record_expansion_statistics: bool,
}

impl InstrumentationParameters {
    fn load_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> Self {
        Self {
            memory_monitoring: MemoryMonitoringParameters::load_from_map(map),
            record_expansion_statistics: didp_mpi::load_bool_from_map(
                map,
                "record_expansion_statistics",
            )
            .unwrap_or(false),
        }
    }
}

impl MemoryMonitoringParameters {
    fn load_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> Self {
        let enabled = didp_mpi::load_bool_from_map(map, "memory_monitoring").unwrap_or(false);
        let interval = map
            .get(&Yaml::String(String::from("memory_monitoring_interval")))
            .map(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_i64().map(|value| value as f64))
                    .unwrap_or_else(|| panic!("memory_monitoring_interval must be numeric"))
            })
            .unwrap_or(1.0);
        assert!(
            interval.is_finite() && interval > 0.0,
            "memory_monitoring_interval must be positive and finite"
        );

        let processes = match map.get(&Yaml::String(String::from("memory_monitoring_processes"))) {
            None => MemoryMonitoringProcesses::Ranks(vec![0]),
            Some(Yaml::String(value)) if value == "all" => MemoryMonitoringProcesses::All,
            Some(Yaml::Array(values)) => MemoryMonitoringProcesses::Ranks(
                values
                    .iter()
                    .map(|value| {
                        let rank = value.as_i64().unwrap_or_else(|| {
                            panic!("memory_monitoring_processes must contain integers")
                        });
                        Rank::try_from(rank).unwrap_or_else(|_| {
                            panic!("memory_monitoring_processes contains an invalid rank")
                        })
                    })
                    .collect(),
            ),
            Some(_) => panic!("memory_monitoring_processes must be `all` or a list of ranks"),
        };

        Self {
            enabled,
            processes,
            interval,
        }
    }

    fn validate(&self, n_processes: Rank) {
        if let MemoryMonitoringProcesses::Ranks(ranks) = &self.processes {
            assert!(
                ranks.iter().all(|rank| (0..n_processes).contains(rank)),
                "memory_monitoring_processes contains a rank outside the MPI communicator"
            );
        }
    }

    fn monitors(&self, rank: Rank) -> bool {
        self.enabled
            && match &self.processes {
                MemoryMonitoringProcesses::All => true,
                MemoryMonitoringProcesses::Ranks(ranks) => ranks.contains(&rank),
            }
    }
}

fn main_with_cost_type_and_hash_function<T, H>(
    universe: Universe,
    model: Model,
    mut parameters: Parameters<T>,
    f_evaluator_type: FEvaluatorType,
    hash_function: H,
    time_keeper: &TimeKeeper,
    instrumentation: InstrumentationParameters,
) where
    T: Numeric + IsFloat + Ord + Display + Hash,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    H: Fn(&HashableSignatureVariables) -> u64,
{
    let InstrumentationParameters {
        memory_monitoring,
        record_expansion_statistics,
    } = instrumentation;
    let (input, evaluators) = didp_mpi::make_input_and_mpi_dual_bound_evaluators(
        model,
        f_evaluator_type,
        parameters.primal_bound,
    );

    let communicator = universe.world();
    memory_monitoring.validate(communicator.size());
    let monitors_memory = memory_monitoring.monitors(communicator.rank());

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
        parameters,
        dual_bound: None,
    };

    if communicator.rank() == 0 {
        println!("Time for initialization: {}s", time_keeper.elapsed_time());
    }

    let mut solver = HdHac::new(input, evaluators, parameters, hash_function, &communicator);
    if communicator.rank() == 0 && memory_monitoring.enabled {
        let layout = solver.search_node_memory_layout();
        println!("Estimated search-node storage layout:");
        println!(
            "  node and state per retained node: {} bytes",
            layout.estimated_node_and_state_bytes
        );
        println!(
            "    node struct: {} bytes (including {}-byte StateInRegistry)",
            layout.search_node_inline_bytes, layout.state_inline_bytes
        );
        println!(
            "    signature struct: {} bytes; state-variable payload: {} bytes",
            layout.signature_variables_inline_bytes, layout.state_variable_payload_bytes
        );
        println!(
            "    estimated state-registry entry: {} bytes",
            layout.state_registry_entry_bytes
        );
        println!(
            "  transition chain per node: {} bytes",
            layout.transition_chain_bytes
        );
        println!(
            "  open-list entries: primary {} bytes; layered {} bytes",
            layout.primary_open_entry_bytes, layout.layered_open_entry_bytes
        );
    }
    if monitors_memory {
        solver.enable_memory_monitoring(memory_monitoring.interval);
    }
    if record_expansion_statistics {
        solver.enable_expansion_statistics();
    }
    let (solution, statistics_list) = solver.search();

    if monitors_memory {
        let filename = format!("memory_statistics_rank_{}.csv", communicator.rank());
        HdHacMemoryStatistics::dump_to_csv(solver.memory_statistics(), &filename).unwrap();
    }
    if let Some(statistics) = solver.expansion_statistics() {
        let filename = format!("expansion_statistics_rank_{}.csv", communicator.rank());
        statistics.dump_to_csv(&filename).unwrap();
        if let Some(statistics) = statistics
            .gather(&communicator, 0, TAG_EXPANSION_STATISTICS)
            .unwrap()
        {
            statistics.dump_to_csv("expansion_statistics.csv").unwrap();
            println!(
                "Expansion statistics: expansion_statistics.csv and expansion_statistics_rank_<rank>.csv"
            );
        }
    }

    if communicator.rank() == 0 {
        didp_mpi::dump_solution(&solution);

        println!(
            "Time to the final solution: {}s",
            time_keeper.elapsed_time()
        );

        didp_mpi::dump_statistics(&statistics_list);
        Statistics::dump_to_csv(&statistics_list, "statistics.csv").unwrap();
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
    let instrumentation = InstrumentationParameters::load_from_map(map);

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
                f_evaluator_type,
                hash_function,
                time_keeper,
                instrumentation,
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
                time_keeper,
                instrumentation,
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

#[cfg(test)]
mod tests {
    use super::*;
    use yaml_rust::YamlLoader;

    #[test]
    fn load_default_memory_monitoring_parameters() {
        let yaml = YamlLoader::load_from_str("{}").unwrap();
        let parameters = MemoryMonitoringParameters::load_from_map(yaml[0].as_hash().unwrap());

        assert!(!parameters.enabled);
        assert_eq!(parameters.interval, 1.0);
        assert!(!parameters.monitors(0));
    }

    #[test]
    fn monitor_only_process_zero_by_default() {
        let yaml = YamlLoader::load_from_str("memory_monitoring: true").unwrap();
        let parameters = MemoryMonitoringParameters::load_from_map(yaml[0].as_hash().unwrap());

        assert!(parameters.monitors(0));
        assert!(!parameters.monitors(1));
        assert_eq!(parameters.interval, 1.0);
    }

    #[test]
    fn load_memory_monitoring_parameters() {
        let yaml = YamlLoader::load_from_str(
            "memory_monitoring: true\nmemory_monitoring_interval: 2\nmemory_monitoring_processes: [1, 2]\n",
        )
        .unwrap();
        let parameters = MemoryMonitoringParameters::load_from_map(yaml[0].as_hash().unwrap());

        assert_eq!(parameters.interval, 2.0);
        assert!(!parameters.monitors(0));
        assert!(parameters.monitors(1));
        assert!(parameters.monitors(2));
    }
}
