use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::data_structure::HashableSignatureVariables;
use mpi::Rank;
use std::fmt::{Debug, Display};
use std::iter::Iterator;
use std::rc::Rc;

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::hash_functions;
use crate::initiation::InitiationResult;
use crate::util;

pub fn compute_hash_values<H, V>(
    hash_function: &mut H,
    nodes: &[(Rc<HashableSignatureVariables>, V)],
    result: &mut Vec<u64>,
) where
    H: FnMut(&HashableSignatureVariables) -> u64,
{
    result.clear();
    result.reserve(nodes.len());

    for node in nodes {
        result.push(hash_function(&node.0));
    }
}

pub fn make_assignment(hash_values: &[u64], n_ranks: Rank, result: &mut Vec<Rank>) {
    result.clear();
    result.reserve(hash_values.len());
    let n_ranks = n_ranks as u64;

    for hash_value in hash_values.iter() {
        result.push((hash_value % n_ranks) as Rank);
    }
}

#[derive(Clone, Debug, Default)]
pub struct AssignemntDistribution {
    pub rank_to_size: Vec<usize>,
    pub max_size: usize,
    pub min_size: usize,
    pub max_rank: Rank,
    pub min_rank: Rank,
}

pub fn compute_assignemnt_distribution<K, V>(
    n_ranks: Rank,
    ranks: &[Rank],
    nodes: &[(K, Vec<V>)],
    result: &mut AssignemntDistribution,
) {
    let n_ranks = n_ranks as usize;
    result.rank_to_size.clear();
    result.rank_to_size.resize(n_ranks, 0);
    result.max_size = 0;
    result.max_rank = 0;

    for (rank, nodes) in ranks.iter().zip(nodes) {
        result.rank_to_size[*rank as usize] += nodes.1.len();

        if result.rank_to_size[*rank as usize] > result.max_size {
            result.max_size = result.rank_to_size[*rank as usize];
            result.max_rank = *rank;
        }
    }

    result.min_size = result.max_size;
    result.min_rank = result.max_rank;

    for (rank, size) in result.rank_to_size.iter().enumerate() {
        if *size < result.min_size {
            result.min_size = *size;
            result.min_rank = rank as Rank;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AahParameters {
    pub max_probability: f64,
    pub step_size: f64,
    pub threshold_ratio_to_average: Option<f64>,
    pub threshold_ratio_to_base: Option<f64>,
}

pub struct AahResult {
    pub masks: Option<Vec<Vec<u32>>>,
    pub probability: Option<f64>,
    pub assignments: Vec<Rank>,
    pub stddev: f64,
}

pub fn aah<T, N, V>(
    model: &Model,
    result: &InitiationResult<T, N, V>,
    n_ranks: Rank,
    parameters: &AahParameters,
) -> AahResult
where
    T: Numeric + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T>,
    V: TransitionInterface + Clone + Default,
{
    let average_size = result.n_nodes as f64 / n_ranks as f64;

    let mut hash_values = Vec::new();
    let mut no_abstraction_assignments = Vec::new();
    let mut distribution = AssignemntDistribution::default();
    distribution.rank_to_size.reserve(n_ranks as usize);

    let mut hash_function = hash_functions::create_fx_hash();
    compute_hash_values(&mut hash_function, &result.nodes, &mut hash_values);
    make_assignment(&hash_values, n_ranks, &mut no_abstraction_assignments);
    compute_assignemnt_distribution(
        n_ranks,
        &no_abstraction_assignments,
        &result.nodes,
        &mut distribution,
    );
    let no_abstraction_stddev = util::compute_stddev(&distribution.rank_to_size);

    if let Some(threshold) = parameters.threshold_ratio_to_average {
        if no_abstraction_stddev / average_size > threshold {
            return AahResult {
                masks: None,
                probability: None,
                assignments: no_abstraction_assignments,
                stddev: no_abstraction_stddev,
            };
        }
    }

    let mut assignments = Vec::with_capacity(no_abstraction_assignments.len());
    let mut p = parameters.max_probability;

    while p >= 0.0 {
        let masks = hash_functions::create_set_masks(model, p);
        let mut hash_function = hash_functions::create_masked_fx_hash(masks.clone());
        compute_hash_values(&mut hash_function, &result.nodes, &mut hash_values);
        make_assignment(&hash_values, n_ranks, &mut assignments);
        compute_assignemnt_distribution(n_ranks, &assignments, &result.nodes, &mut distribution);
        let stddev = util::compute_stddev(&distribution.rank_to_size);

        let mut is_good = true;

        if let Some(threshold) = parameters.threshold_ratio_to_average {
            if stddev / average_size > threshold {
                is_good = false;
            }
        }

        if let Some(threshold) = parameters.threshold_ratio_to_base {
            if stddev / no_abstraction_stddev > threshold {
                is_good = false;
            }
        }

        if is_good {
            return AahResult {
                masks: Some(masks),
                probability: Some(p),
                assignments,
                stddev,
            };
        }

        p -= parameters.step_size;
    }

    AahResult {
        masks: None,
        probability: None,
        assignments: no_abstraction_assignments,
        stddev: no_abstraction_stddev,
    }
}
