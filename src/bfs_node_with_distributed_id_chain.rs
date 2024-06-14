use std::fmt::Display;

use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure::StateInformation;

use crate::distributed_id_chain::GeRctDistributedTransitionIdChain;

pub trait BfsNodeWithDistributedIdChain<T>:
    Ord + StateInformation<T> + GeRctDistributedTransitionIdChain
where
    T: Numeric + Display,
{
    fn ordered_by_bound() -> bool;
}

#[derive(Clone)]
pub struct NodeGenerationResult<N> {
    pub node: Option<N>,
    pub is_pruned_by_bound: bool,
    pub n_dominated_before_closed: usize,
    pub n_dominated_after_closed: usize,
}

impl<N> Default for NodeGenerationResult<N> {
    fn default() -> Self {
        Self {
            node: None,
            is_pruned_by_bound: false,
            n_dominated_before_closed: 0,
            n_dominated_after_closed: 0,
        }
    }
}
