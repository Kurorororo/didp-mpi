use crate::{distributed_id_chain::GetDistributedTransitionIdChain, node_data_type::NodeDatatype};
use dypdl::prelude::*;
use dypdl_heuristic_search::search_algorithm::data_structure::{
    HashableSignatureVariables, StateWithHashableSignatureVariables,
};
use mpi::Rank;

pub trait NodeMessage<T>: NodeDatatype<T, S = Self> + GetDistributedTransitionIdChain {
    fn state(&self) -> &StateWithHashableSignatureVariables;

    fn cost(&self, model: &Model) -> T;

    fn bound(&self, model: &Model) -> Option<T>;

    fn signature(&self) -> &HashableSignatureVariables;

    fn set_parent_rank(&self, parent_rank: Rank);
}
