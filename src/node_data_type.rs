use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::data_structure::HashableSignatureVariables;
use mpi::{
    datatype::{DatatypeRef, UserDatatype},
    Address, Count, Rank,
};

use crate::{
    distributed_id_chain::DistributedTransitionIdChain, is_float::IsFloat,
    state_serializer::StateSerializer,
};

pub trait NodeDatatype<T: IsFloat> {
    fn get_total_size(serializer: &StateSerializer) -> usize;

    fn get_datatype_blocklengths(serializer: &StateSerializer) -> Vec<Count>;

    fn get_datatype_displacements(serializer: &StateSerializer) -> Vec<Address>;

    fn get_datatype_types(serializer: &StateSerializer) -> Vec<DatatypeRef<'static>>;

    fn create_data_type(serializer: &StateSerializer) -> UserDatatype {
        let blocklengths = Self::get_datatype_blocklengths(serializer);
        let displacements = Self::get_datatype_displacements(serializer);
        let types = Self::get_datatype_types(serializer);

        UserDatatype::structured(&blocklengths, &displacements, &types)
    }

    fn serialize_to(&self, serializer: &StateSerializer, buffer: &mut [u8]);

    fn deserialize(serializer: &StateSerializer, buffer: &[u8]) -> Self;

    fn get_signature(&self) -> &HashableSignatureVariables;

    fn get_bound(model: &Model, serializer: &StateSerializer, buffer: &[u8]) -> Option<T>;

    fn set_parent_rank(&self, parent_rank: Rank);

    fn get_solution_cost_and_suffix<'a, V, B>(
        &self,
        model: &Model,
        suffix: &'a [V],
        base_cost_evaluator: B,
    ) -> Option<(T, &'a [V])>
    where
        T: Numeric + Ord,
        V: TransitionInterface,
        B: FnMut(T, T) -> T;

    fn get_distributed_transition_id_chain(&self) -> &DistributedTransitionIdChain;
}
