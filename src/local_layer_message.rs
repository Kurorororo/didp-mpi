use dypdl::{prelude::*, variable_type::Numeric};
use mpi::{
    datatype::{SystemDatatype, UserDatatype},
    traits::*,
    Address, Rank, Tag,
};

use crate::is_float::IsFloat;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalLayerMessage<T> {
    pub pruned: bool,
    pub is_empty: bool,
    pub time_out: bool,
    pub bound: Option<T>,
    pub cost: Option<T>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct LocalLayerMessageForSend<T>([bool; 5], [T; 2]);

unsafe impl<T> Equivalence for LocalLayerMessageForSend<T>
where
    T: Equivalence<Out = SystemDatatype>,
{
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[5, 2],
            &[
                memoffset::offset_of!(LocalLayerMessageForSend<T>, 0) as Address,
                memoffset::offset_of!(LocalLayerMessageForSend<T>, 1) as Address,
            ],
            &[bool::equivalent_datatype(), T::equivalent_datatype()],
        )
    }
}

impl<T, U> From<LocalLayerMessage<T>> for LocalLayerMessageForSend<U>
where
    T: Numeric,
    U: Numeric,
{
    fn from(message: LocalLayerMessage<T>) -> Self {
        let bound = message
            .bound
            .map_or_else(|| U::default(), |bound| U::from(bound));
        let cost = message
            .cost
            .map_or_else(|| U::default(), |cost| U::from(cost));

        Self(
            [
                message.pruned,
                message.is_empty,
                message.time_out,
                message.bound.is_some(),
                message.cost.is_some(),
            ],
            [bound, cost],
        )
    }
}

impl<T, U> From<LocalLayerMessageForSend<T>> for LocalLayerMessage<U>
where
    T: Numeric,
    U: Numeric,
{
    fn from(message: LocalLayerMessageForSend<T>) -> Self {
        let bound = if message.0[3] {
            Some(U::from(message.1[0]))
        } else {
            None
        };
        let cost = if message.0[4] {
            Some(U::from(message.1[1]))
        } else {
            None
        };

        Self {
            pruned: message.0[0],
            is_empty: message.0[1],
            time_out: message.0[2],
            bound,
            cost,
        }
    }
}

impl<T: IsFloat> LocalLayerMessage<T> {
    pub fn send<C: Communicator>(&self, communicator: &C, destination_rank: Rank, tag: Tag) {
        let destination = communicator.process_at_rank(destination_rank);

        if T::is_float() {
            let message = LocalLayerMessageForSend::<Continuous>::from(self.clone());
            destination.buffered_send_with_tag(&message, tag);
        } else {
            let message = LocalLayerMessageForSend::<Integer>::from(self.clone());
            destination.buffered_send_with_tag(&message, tag);
        }
    }

    pub fn receive<C: Communicator>(communicator: &C, source_rank: Rank, tag: Tag) -> Self {
        let source = communicator.process_at_rank(source_rank);

        if T::is_float() {
            let mut message = LocalLayerMessageForSend::<Continuous>::default();
            source.receive_into_with_tag(&mut message, tag);
            Self::from(message)
        } else {
            let mut message = LocalLayerMessageForSend::<Integer>::default();
            source.receive_into_with_tag(&mut message, tag);
            Self::from(message)
        }
    }
}
