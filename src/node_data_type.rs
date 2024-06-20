use dypdl::prelude::*;
use mpi::{
    datatype::{DatatypeRef, UserDatatype},
    Address, Count,
};

use crate::state_serializer::StateSerializer;

pub trait NodeDatatype<T> {
    type S;

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

    fn deserialize(serializer: &StateSerializer, buffer: &[u8]) -> Self::S;

    fn get_bound_from_buffer(
        model: &Model,
        serializer: &StateSerializer,
        buffer: &[u8],
    ) -> Option<T>;
}
