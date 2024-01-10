use std::fmt::Display;

use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure::StateInformation;

use crate::distributed_id_chain::GetDistributedTransitionIdChain;

pub trait BfsNodeWithDistributedIdChain<T>: Ord + StateInformation<T> + GetDistributedTransitionIdChain
where
    T: Numeric + Display,
{
    fn ordered_by_bound() -> bool;
}
