use didp_mpi::{
    AdditionalCommonParameters, DistributedFNode, DistributedFNodeMessage, HashType, IsFloat,
    Statistics,
};
use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{
    prelude::*,
    variable_type::{Numeric, OrderedContinuous},
};
use dypdl_heuristic_search::{
    search_algorithm::{
        data_structure::HashableSignatureVariables, Cabs, SearchInput, SuccessorGenerator,
        TransitionWithId,
    },
    BeamSearchParameters, CabsParameters, FEvaluatorType, Search,
};
use mpi::{environment::Universe, traits::*};
use std::fmt::{Debug, Display};
use std::fs::{self, OpenOptions};
use std::io::Write;
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
    mut parameters: CabsParameters<T>,
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
        parameters.beam_search_parameters.parameters.primal_bound,
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

    if communicator.rank() == 0 {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("history.csv")
            .unwrap();
        file.write_all(String::from("").as_bytes()).unwrap();
    }

    let mut statistics = Statistics::default();
    let mut root_rank = None;

    let beam_search = |input: &SearchInput<_, _>, parameters| {
        let (solution, goal_rank, tmp_statistics) = didp_mpi::hd_beam_search2(
            input,
            &transition_evaluator,
            base_cost_evaluator,
            parameters,
            &hash_function,
            &communicator,
            0,
        );
        statistics += tmp_statistics;

        if goal_rank == Some(communicator.rank()) && solution.cost.is_some() {
            didp_mpi::write_solution(&solution, "solution.yaml");

            let mut file = OpenOptions::new().append(true).open("history.csv").unwrap();
            let line = format!(
                "{}, {}, {}, {}\n",
                solution.time,
                solution.cost.unwrap(),
                solution.expanded,
                solution.generated
            );
            file.write_all(line.as_bytes()).unwrap();
        }

        if goal_rank.is_some() {
            root_rank = goal_rank;
        }

        solution
    };

    parameters.beam_search_parameters.parameters.quiet |= communicator.rank() != 0;

    let mut solver = Cabs::<_, _, _, _>::new(input, beam_search, parameters);
    let mut solution = solver.search().unwrap();

    let root_rank = root_rank.unwrap_or(0);
    let is_root = communicator.rank() == root_rank;
    let statistics_list = statistics.gather(&communicator, root_rank, is_root);

    if is_root {
        solution.expanded = statistics_list.iter().map(|s| s.expanded).sum();
        solution.generated = statistics_list.iter().map(|s| s.generated).sum();
        didp_mpi::dump_solution(&solution);
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
    let (parameters, additional_parameters) = load_config_from_file::<T>(config_filename);

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
                f_evaluator_type,
                hash_function,
            );
        }
    }
}

fn load_config_from_file<T>(filename: &str) -> (CabsParameters<T>, AdditionalCommonParameters)
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
    let parameters = didp_mpi::load_parameters_from_map::<T>(map);
    let additional_common_parameters = AdditionalCommonParameters::load_from_map(map);

    let beam_size = match map.get(&yaml_rust::Yaml::from_str("initial_beam_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            panic!(
                "expected Integer for `initial_beam_size`, but found `{:?}`",
                value
            )
        }
        None => 1,
    };
    let keep_all_layers = match map.get(&yaml_rust::Yaml::from_str("keep_all_layers")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            panic!(
                "expected Boolean for `keep_all_layers`, but found `{:?}`",
                value
            )
        }
    };
    let max_beam_size = match map.get(&yaml_rust::Yaml::from_str("max_beam_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        Some(value) => {
            panic!(
                "expected Integer for `max_beam_size`, but found `{:?}`",
                value
            )
        }
        None => None,
    };
    let beam_search_parameters = BeamSearchParameters {
        parameters,
        beam_size,
        keep_all_layers,
    };
    let parameters = CabsParameters {
        max_beam_size,
        beam_search_parameters,
    };

    (parameters, additional_common_parameters)
}

fn main() {
    let universe = mpi::initialize().unwrap();

    let mut args = std::env::args();
    args.next();
    let model = didp_mpi::read_model(&mut args);
    let config_filename = args.next().expect("Config filename is not specified");

    match model.cost_type {
        CostType::Integer => main_with_cost_type::<Integer>(universe, model, &config_filename),
        CostType::Continuous => {
            main_with_cost_type::<OrderedContinuous>(universe, model, &config_filename)
        }
    }
}
