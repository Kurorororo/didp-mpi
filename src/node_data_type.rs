use dypdl::prelude::*;
use mpi::{
    datatype::{DatatypeRef, UserDatatype},
    Address, Count,
};

use crate::{is_float::IsFloat, state_serializer::StateSerializer};

pub trait NodeDatatype<T: IsFloat> {
    fn get_total_size(serializer: &StateSerializer) -> usize;

    fn get_datatype_blocklengths(serializer: &StateSerializer) -> Vec<Count>;

    fn get_datatype_displacement(serializer: &StateSerializer) -> Vec<Address>;

    fn get_datatype_types(serializer: &StateSerializer) -> Vec<DatatypeRef<'static>>;

    fn create_data_type(serializer: &StateSerializer) -> UserDatatype {
        let blocklengths = Self::get_datatype_blocklengths(serializer);
        let displacements = Self::get_datatype_displacement(serializer);
        let types = Self::get_datatype_types(serializer);

        UserDatatype::structured(&blocklengths, &displacements, &types)
    }

    fn serialize_to(&self, serializer: &StateSerializer, buffer: &mut [u8]);

    fn deserialize(serializer: &StateSerializer, buffer: &[u8]) -> Self;

    fn get_bound(model: &Model, serializer: &StateSerializer, buffer: &[u8]) -> Option<T>;
}
