use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::data_structure;
use mpi::{
    datatype::{MutView, UserDatatype, View},
    Rank, Tag,
};
use mpi::{traits::*, Address};
use std::marker::PhantomData;
use std::mem;
use std::rc::Rc;
use zerocopy::{AsBytes, FromBytes};

use crate::is_float::IsFloat;
use crate::node_data_type::NodeDatatype;
use crate::state_serializer::StateSerializer;
use crate::timestamped_communicator::TimestampedCommunicator;

pub struct NodeCommunicator<'a, C, M, T> {
    model: Rc<Model>,
    communicator: &'a C,
    tag: Tag,
    state_serializer: StateSerializer,
    user_datatype: UserDatatype,
    tmp_buffer: Vec<u8>,
    _phantom: PhantomData<(M, T)>,
}

impl<'a, C, M, T> NodeCommunicator<'a, C, M, T>
where
    C: Communicator,
    M: NodeDatatype<T, S = M>,
    T: Numeric + IsFloat,
{
    pub fn new(communicator: &'a C, tag: Tag, model: Rc<Model>) -> Self {
        let state_serializer = StateSerializer::with_model(&model);
        let user_datatype = M::create_data_type(&state_serializer);
        let tmp_buffer = vec![0; M::get_total_size(&state_serializer)];

        Self {
            model,
            communicator,
            tag,
            state_serializer,
            user_datatype,
            tmp_buffer,
            _phantom: PhantomData,
        }
    }

    pub fn send<N>(&mut self, destination_rank: Rank, node: N)
    where
        N: NodeDatatype<T, S = M>,
    {
        node.serialize_to(&self.state_serializer, &mut self.tmp_buffer);
        let destination = self.communicator.process_at_rank(destination_rank);
        let v = unsafe { View::with_count_and_datatype(&self.tmp_buffer, 1, &self.user_datatype) };
        destination.buffered_send_with_tag(&v, self.tag);
    }

    pub fn receive(&mut self, source_rank: Rank, primal_bound: Option<T>) -> Option<M> {
        let source = self.communicator.process_at_rank(source_rank);
        let mut v = unsafe {
            MutView::with_count_and_datatype(&mut self.tmp_buffer, 1, &self.user_datatype)
        };
        source.receive_into_with_tag(&mut v, self.tag);

        if let Some(bound) =
            M::get_bound_from_buffer(&self.model, &self.state_serializer, &self.tmp_buffer)
        {
            if data_structure::exceed_bound(&self.model, bound, primal_bound) {
                return None;
            }
        }

        Some(M::deserialize(&self.state_serializer, &self.tmp_buffer))
    }
}

pub struct TimeStampedNodeCommunicator<'a, C, M, T> {
    model: Rc<Model>,
    communicator: TimestampedCommunicator<'a, C>,
    state_serializer: StateSerializer,
    tmp_buffer: Vec<u8>,
    _phantom: PhantomData<(M, T)>,
}

impl<'a, C, M, T> TimeStampedNodeCommunicator<'a, C, M, T>
where
    C: Communicator,
    M: NodeDatatype<T, S = M>,
    T: Numeric + IsFloat,
{
    pub fn new(
        communicator: &'a C,
        tag: Tag,
        tag_termination_detection: Tag,
        model: Rc<Model>,
    ) -> Self {
        let state_serializer = StateSerializer::with_model(&model);

        let blocklengths = M::get_datatype_blocklengths(&state_serializer);
        let displacements = M::get_datatype_displacements(&state_serializer);
        let types = M::get_datatype_types(&state_serializer);
        let offset = M::get_total_size(&state_serializer);

        let communicator = TimestampedCommunicator::new(
            communicator,
            tag,
            tag_termination_detection,
            &blocklengths,
            &displacements,
            &types,
            offset as Address,
        );

        let tmp_buffer = vec![0; offset];

        Self {
            model,
            communicator,
            state_serializer,
            tmp_buffer,
            _phantom: PhantomData,
        }
    }

    pub fn send<N>(&mut self, destination_rank: Rank, node: &N)
    where
        N: NodeDatatype<T, S = M>,
    {
        node.serialize_to(&self.state_serializer, &mut self.tmp_buffer);
        self.communicator
            .send(&mut self.tmp_buffer, destination_rank);
    }

    pub fn receive(&mut self, source_rank: Rank, primal_bound: Option<T>) -> Option<M> {
        self.communicator
            .receive_into(&mut self.tmp_buffer, source_rank);

        if let Some(bound) =
            M::get_bound_from_buffer(&self.model, &self.state_serializer, &self.tmp_buffer)
        {
            if data_structure::exceed_bound(&self.model, bound, primal_bound) {
                return None;
            }
        }

        let node = M::deserialize(&self.state_serializer, &self.tmp_buffer);

        Some(node)
    }

    pub fn receive_and_discard(&mut self, source_rank: Rank, dual_bound: Option<T>) -> Option<T> {
        self.communicator
            .receive_into(&mut self.tmp_buffer, source_rank);

        if let Some(bound) =
            M::get_bound_from_buffer(&self.model, &self.state_serializer, &self.tmp_buffer)
        {
            if data_structure::exceed_bound(&self.model, bound, dual_bound) {
                None
            } else {
                Some(bound)
            }
        } else {
            None
        }
    }

    pub fn initiate_termination(&mut self, destination_rank: Rank) {
        self.communicator.initiate_termination(destination_rank);
    }

    pub fn receive_termination_detection_and_forward(
        &mut self,
        source_rank: Rank,
        destination_rank: Rank,
        local_invalid: bool,
    ) -> Option<bool> {
        self.communicator.receive_termination_detection_and_forward(
            source_rank,
            destination_rank,
            local_invalid,
        )
    }
}

pub struct TimeStampedNodeDepthCommunicator<'a, C, M, T> {
    model: Rc<Model>,
    communicator: TimestampedCommunicator<'a, C>,
    state_serializer: StateSerializer,
    offset: usize,
    tmp_buffer: Vec<u8>,
    _phantom: PhantomData<(M, T)>,
}

impl<'a, C, M, T> TimeStampedNodeDepthCommunicator<'a, C, M, T>
where
    C: Communicator,
    M: NodeDatatype<T, S = M>,
    T: Numeric + IsFloat,
{
    pub fn new(
        communicator: &'a C,
        tag: Tag,
        tag_termination_detection: Tag,
        model: Rc<Model>,
    ) -> Self {
        let state_serializer = StateSerializer::with_model(&model);

        let mut blocklengths = M::get_datatype_blocklengths(&state_serializer);
        let mut displacements = M::get_datatype_displacements(&state_serializer);
        let mut types = M::get_datatype_types(&state_serializer);
        let offset = M::get_total_size(&state_serializer);

        blocklengths.push(1);
        displacements.push(offset as Address);
        types.push(usize::equivalent_datatype());
        let total_size = offset + mem::size_of::<usize>();

        let communicator = TimestampedCommunicator::new(
            communicator,
            tag,
            tag_termination_detection,
            &blocklengths,
            &displacements,
            &types,
            total_size as Address,
        );

        let tmp_buffer = vec![0; total_size];

        Self {
            model,
            communicator,
            state_serializer,
            offset,
            tmp_buffer,
            _phantom: PhantomData,
        }
    }

    pub fn send<N>(&mut self, destination_rank: Rank, node: &N, depth: usize)
    where
        N: NodeDatatype<T, S = M>,
    {
        node.serialize_to(&self.state_serializer, &mut self.tmp_buffer);
        self.tmp_buffer[self.offset..self.offset + mem::size_of::<usize>()]
            .copy_from_slice(depth.as_bytes());

        self.communicator
            .send(&mut self.tmp_buffer, destination_rank);
    }

    pub fn receive(&mut self, source_rank: Rank, primal_bound: Option<T>) -> Option<(M, usize)> {
        self.communicator
            .receive_into(&mut self.tmp_buffer, source_rank);

        if let Some(bound) =
            M::get_bound_from_buffer(&self.model, &self.state_serializer, &self.tmp_buffer)
        {
            if data_structure::exceed_bound(&self.model, bound, primal_bound) {
                return None;
            }
        }

        let node = M::deserialize(&self.state_serializer, &self.tmp_buffer);
        let depth =
            usize::read_from(&self.tmp_buffer[self.offset..self.offset + mem::size_of::<usize>()])
                .unwrap();

        Some((node, depth))
    }

    pub fn receive_and_discard(&mut self, source_rank: Rank, dual_bound: Option<T>) -> Option<T> {
        self.communicator
            .receive_into(&mut self.tmp_buffer, source_rank);

        if let Some(bound) =
            M::get_bound_from_buffer(&self.model, &self.state_serializer, &self.tmp_buffer)
        {
            if data_structure::exceed_bound(&self.model, bound, dual_bound) {
                None
            } else {
                Some(bound)
            }
        } else {
            None
        }
    }

    pub fn initiate_termination(&mut self, destination_rank: Rank) {
        self.communicator.initiate_termination(destination_rank);
    }

    pub fn receive_termination_detection_and_forward(
        &mut self,
        source_rank: Rank,
        destination_rank: Rank,
        local_invalid: bool,
    ) -> Option<bool> {
        self.communicator.receive_termination_detection_and_forward(
            source_rank,
            destination_rank,
            local_invalid,
        )
    }
}
