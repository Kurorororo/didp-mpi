use didp_mpi::{
    AssignemntDistribution, DistributedFNode, FNodeEvaluators, InitiationParameters, IsFloat,
};
use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{
    prelude::*,
    variable_type::{Numeric, OrderedContinuous},
};
use dypdl_heuristic_search::{
    search_algorithm::{util::TimeKeeper, SearchInput, SuccessorGenerator, TransitionWithId},
    FEvaluatorType,
};
use linked_hash_map::LinkedHashMap;
use mpi::Rank;
use std::fmt::{Debug, Display};
use std::fs;
use std::hash::Hash;
use std::rc::Rc;
use std::str::FromStr;
use yaml_rust::{Yaml, YamlLoader};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn load_initiation_parameters_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> InitiationParameters {
    let time_limit = map
        .get(&yaml_rust::Yaml::from_str("time_limit"))
        .map(|value| didp_yaml::util::get_numeric(value).unwrap());
    let quiet = match map.get(&yaml_rust::Yaml::from_str("quiet")) {
        Some(Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            panic!("expected Boolean, but found `{:?}`", value)
        }
    };
    let node_limit = match map.get(&Yaml::from_str("node_limit")) {
        Some(Yaml::Integer(value)) => Some(*value as usize),
        None => Some(1000000),
        value => {
            panic!("expected Integer for `node_limit`, but found `{:?}`", value)
        }
    };

    InitiationParameters {
        time_limit,
        quiet,
        node_limit,
    }
}

fn load_config_from_file(
    filename: &str,
) -> (InitiationParameters, FEvaluatorType, Rank, Option<f64>) {
    let config = fs::read_to_string(filename).unwrap_or_else(|e| {
        panic!("Couldn't read a config file: {:?}", e);
    });
    let config = YamlLoader::load_from_str(&config).unwrap_or_else(|e| {
        panic!("Config file must be in YAML format: {:?}", e);
    });
    assert_eq!(config.len(), 1);
    let yaml = &config[0];
    let map = yaml.as_hash().expect("Yaml file is not a hash");
    let parameters = load_initiation_parameters_from_map(map);

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

    let n_ranks = match map.get(&Yaml::from_str("n_ranks")) {
        Some(Yaml::Integer(value)) => *value as Rank,
        None => 1,
        value => {
            panic!(
                "expected Integer for `initial_registry_capacity`, but found `{:?}`",
                value
            )
        }
    };

    let ratio_threshold = map
        .get(&Yaml::String("ratio_threshold".into()))
        .map(|x| x.as_f64().expect("ratio_threshold must be a float"));

    (parameters, f_evaluator_type, n_ranks, ratio_threshold)
}

fn compute_variance(values: &[usize]) -> f64 {
    let n = values.len() as f64;
    let sum = values.iter().sum::<usize>() as f64;
    let mean = sum / n;
    let variance = values
        .iter()
        .map(|x| (*x as f64 - mean).powi(2))
        .sum::<f64>()
        / n;

    variance
}

fn compute_stddev(values: &[usize]) -> f64 {
    compute_variance(values).sqrt()
}

fn main_with_cost_type<T>(
    model: Model,
    parameters: InitiationParameters,
    f_evaluator_type: FEvaluatorType,
    n_ranks: Rank,
    ratio_threshold: Option<f64>,
) where
    T: Numeric + Ord + Display + Hash + IsFloat + 'static,
    CostToDump: From<T>,
    <T as FromStr>::Err: Debug,
{
    let time_keepr = TimeKeeper::default();

    let model = Rc::new(model);
    let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);
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
    let node = DistributedFNode::<_>::generate_root_node(
        model.target.clone(),
        cost,
        &model,
        &h_evaluator,
        &f_evaluator,
        None,
    );
    let input = SearchInput {
        node,
        generator,
        solution_suffix: &[],
    };
    let successor_evaluator =
        move |state, cost, transition: &_, chain: &_, registry: &mut _, primal_bound| {
            let evaluators = FNodeEvaluators {
                h: &h_evaluator,
                f: &f_evaluator,
            };

            DistributedFNode::insert_node(
                state,
                cost,
                transition,
                chain,
                registry,
                evaluators,
                primal_bound,
            )
        };

    let result =
        didp_mpi::hac_initiator(input, successor_evaluator, base_cost_evaluator, parameters);

    println!(
        "Generated {} nodes with {} seconds.",
        result.n_nodes,
        time_keepr.elapsed_time(),
    );

    if let Some(cost) = result.solution.cost {
        println!("Initial solution cost: {}", cost);
    }

    if result.solution.is_optimal {
        println!("Optimal solution is found.");
    } else if result.solution.is_infeasible {
        println!("Infeasibility is proved.");
    } else {
        let mut hash_values = Vec::default();
        let mut assignments = Vec::default();
        let mut distribution = AssignemntDistribution::default();
        distribution.rank_to_size.reserve(n_ranks as usize);

        let mut hash_function = didp_mpi::create_fx_hash();
        didp_mpi::compute_hash_values(&mut hash_function, &result.nodes, &mut hash_values);
        didp_mpi::make_assignment(&hash_values, n_ranks, &mut assignments);
        didp_mpi::compute_assignemnt_distribution(
            n_ranks,
            &assignments,
            &result.nodes,
            &mut distribution,
        );
        let no_abstraction_stddev = compute_stddev(&distribution.rank_to_size);

        println!("No abstraction, stddev = {}", no_abstraction_stddev);

        for p in [0.3, 0.2, 0.1] {
            let table = didp_mpi::create_abstract_bytewise_random_table(&model, p);
            let mut hash_function = didp_mpi::create_bytewise_zobrist_hash(table);
            didp_mpi::compute_hash_values(&mut hash_function, &result.nodes, &mut hash_values);
            didp_mpi::make_assignment(&hash_values, n_ranks, &mut assignments);
            didp_mpi::compute_assignemnt_distribution(
                n_ranks,
                &assignments,
                &result.nodes,
                &mut distribution,
            );
            let stddev = compute_stddev(&distribution.rank_to_size);

            println!("p = {}, stddev = {}", p, stddev);

            if let Some(ratio_threshold) = ratio_threshold {
                if stddev < ratio_threshold * no_abstraction_stddev {
                    break;
                }
            }
        }

        println!("Terminated with {} seconds", time_keepr.elapsed_time());
    }
}

fn main() {
    let mut args = std::env::args();
    args.next();
    let model = didp_mpi::read_model(&mut args);
    let config_filename = args.next().expect("Config filename is not specified");
    let (parameters, f_evaluator_type, n_ranks, ratio_threshold) =
        load_config_from_file(&config_filename);

    match model.cost_type {
        CostType::Integer => main_with_cost_type::<Integer>(
            model,
            parameters,
            f_evaluator_type,
            n_ranks,
            ratio_threshold,
        ),
        CostType::Continuous => main_with_cost_type::<OrderedContinuous>(
            model,
            parameters,
            f_evaluator_type,
            n_ranks,
            ratio_threshold,
        ),
    }
}
