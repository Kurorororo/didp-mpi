use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::{self, HashableSignatureVariables, StateWithHashableSignatureVariables},
    SearchInput, Solution, StateInRegistry, StateRegistry, TransitionWithId,
};
use mpi::{datatype::UserDatatype, topology::SimpleCommunicator, traits::*, Address, Rank, Tag};
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;
use std::str::FromStr;
use std::{cmp, mem};
use std::{
    fmt::{Debug, Display},
    vec,
};
use zerocopy::{AsBytes, FromBytes};

use crate::open_list;
use crate::state_serializer::StateSerializer;
use crate::timestamped_communicator::TimestampedUserCommunicator;
use crate::{
    bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain, KeyValueStatistics,
};
use crate::{bfs_node_with_distributed_id_chain::NodeGenerationResult, is_float::IsFloat};
use crate::{
    distributed_id_chain::DistributedTransitionIdChain,
    mpi_anytime_search::{
        MpiAnytimeSearch, MpiAnytimeSearchEvaluators, MpiAnytimeSearchParameters,
    },
};
use crate::{layered::Layered, node_data_type::NodeDatatype};
use crate::{node_message::NodeMessage, statistics::Statistics};

struct Hdbs3NodeCommunicator<'a, C, M, T> {
    model: Rc<Model>,
    communicator: TimestampedUserCommunicator<'a, C>,
    tag: Tag,
    tag_n_sent: Tag,
    state_serializer: StateSerializer,
    node_datatype: UserDatatype,
    sent_all_datatype: UserDatatype,
    offset_for_depth: usize,
    offset_for_timestamp: usize,
    tmp_buffer: Vec<u8>,
    _phantom: PhantomData<(M, T)>,
}

impl<'a, C, M, T> Hdbs3NodeCommunicator<'a, C, M, T>
where
    C: Communicator,
    M: NodeDatatype<T, S = M>,
    T: Numeric + IsFloat,
{
    fn new(
        communicator: &'a C,
        tag: Tag,
        tag_n_sent: Tag,
        tag_termination_detection: Tag,
        model: Rc<Model>,
    ) -> Self {
        let state_serializer = StateSerializer::with_model(&model);

        let mut blocklengths = M::get_datatype_blocklengths(&state_serializer);
        let mut displacements = M::get_datatype_displacements(&state_serializer);
        let mut types = M::get_datatype_types(&state_serializer);
        let offset_for_depth = M::get_total_size(&state_serializer);

        blocklengths.push(1);
        displacements.push(offset_for_depth as Address);
        types.push(usize::equivalent_datatype());

        let offset_for_timestamp = offset_for_depth + mem::size_of::<usize>();
        blocklengths.push(1);
        displacements.push(offset_for_timestamp as Address);
        types.push(usize::equivalent_datatype());

        let node_datatype = UserDatatype::structured(&blocklengths, &displacements, &types);
        let sent_all_datatype = UserDatatype::structured(
            &[1, 2],
            &[0, mem::size_of::<i32>() as Address],
            &[i32::equivalent_datatype(), usize::equivalent_datatype()],
        );

        let communicator =
            TimestampedUserCommunicator::new(communicator, tag_termination_detection);

        let total_size = offset_for_timestamp + mem::size_of::<usize>();
        let tmp_buffer = vec![0; total_size];

        Self {
            model,
            communicator,
            tag,
            tag_n_sent,
            state_serializer,
            node_datatype,
            sent_all_datatype,
            offset_for_depth,
            offset_for_timestamp,
            tmp_buffer,
            _phantom: PhantomData,
        }
    }

    fn serialize_sent_all_message(n_sent: i32, depth: usize, buffer: &mut [u8]) {
        buffer[..mem::size_of::<i32>()].copy_from_slice(n_sent.as_bytes());
        buffer[mem::size_of::<i32>()..mem::size_of::<i32>() + mem::size_of::<usize>()]
            .copy_from_slice(depth.as_bytes());
    }

    fn deserialize_sent_all_message(buffer: &[u8]) -> (i32, usize) {
        let n_sent = i32::read_from(&buffer[..mem::size_of::<i32>()]).unwrap();
        let depth = usize::read_from(
            &buffer[mem::size_of::<i32>()..mem::size_of::<i32>() + mem::size_of::<usize>()],
        )
        .unwrap();

        (n_sent, depth)
    }

    fn send<N>(&mut self, destination_rank: Rank, node: &N, depth: usize)
    where
        N: NodeDatatype<T, S = M>,
    {
        node.serialize_to(&self.state_serializer, &mut self.tmp_buffer);
        self.tmp_buffer[self.offset_for_depth..self.offset_for_depth + mem::size_of::<usize>()]
            .copy_from_slice(depth.as_bytes());

        self.communicator.buffered_send_with_tag(
            &mut self.tmp_buffer,
            self.offset_for_timestamp,
            destination_rank,
            self.tag,
            &self.node_datatype,
        );
    }

    fn get_depth(&self) -> usize {
        usize::read_from(
            &self.tmp_buffer
                [self.offset_for_depth..self.offset_for_depth + mem::size_of::<usize>()],
        )
        .unwrap()
    }

    fn receive(
        &mut self,
        source_rank: Rank,
        primal_bound: Option<T>,
        depth_bound: usize,
    ) -> (Option<M>, Option<T>, usize) {
        self.communicator.receive_into_with_tag(
            &mut self.tmp_buffer,
            self.offset_for_timestamp,
            source_rank,
            self.tag,
            &self.node_datatype,
        );
        let depth = self.get_depth();
        let bound = M::get_bound_from_buffer(&self.model, &self.state_serializer, &self.tmp_buffer);

        if depth < depth_bound {
            return (None, bound, depth);
        }

        if let Some(bound) = bound {
            if data_structure::exceed_bound(&self.model, bound, primal_bound) {
                return (None, Some(bound), depth);
            }
        }

        let node = M::deserialize(&self.state_serializer, &self.tmp_buffer);

        (Some(node), bound, depth)
    }

    fn receive_and_discard(
        &mut self,
        source_rank: Rank,
        dual_bound: Option<T>,
    ) -> (Option<T>, usize) {
        self.communicator.receive_into_with_tag(
            &mut self.tmp_buffer,
            self.offset_for_timestamp,
            source_rank,
            self.tag,
            &self.node_datatype,
        );
        let depth = self.get_depth();

        if let Some(bound) =
            M::get_bound_from_buffer(&self.model, &self.state_serializer, &self.tmp_buffer)
        {
            if data_structure::exceed_bound(&self.model, bound, dual_bound) {
                (None, depth)
            } else {
                (Some(bound), depth)
            }
        } else {
            (None, depth)
        }
    }

    fn send_n_sent(&mut self, destination_rank: Rank, n_sent: i32, depth: usize) {
        let mut buffer = [0; mem::size_of::<i32>() + 2 * mem::size_of::<usize>()];
        Self::serialize_sent_all_message(n_sent, depth, &mut buffer);

        self.communicator.buffered_send_with_tag(
            &mut buffer,
            mem::size_of::<i32>() + mem::size_of::<usize>(),
            destination_rank,
            self.tag_n_sent,
            &self.sent_all_datatype,
        );
    }

    fn receive_n_sent(&mut self, source_rank: Rank) -> (i32, usize) {
        let mut buffer = [0; mem::size_of::<i32>() + 2 * mem::size_of::<usize>()];
        self.communicator.receive_into_with_tag(
            &mut buffer,
            mem::size_of::<i32>() + mem::size_of::<usize>(),
            source_rank,
            self.tag_n_sent,
            &self.sent_all_datatype,
        );

        Self::deserialize_sent_all_message(&buffer)
    }

    fn initiate_termination(&mut self, destination_rank: Rank) {
        self.communicator.initiate_termination(destination_rank);
    }

    fn receive_termination_detection_and_forward(
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

pub struct Hdbs3<'a, T, N, M, L, R, B, F, V = Transition>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    V: TransitionInterface + Clone + Default,
{
    model: Rc<Model>,
    search: MpiAnytimeSearch<'a, T, N, M, L, R, B, F, V>,
    communicator: &'a SimpleCommunicator,
    node_communicator: Hdbs3NodeCommunicator<'a, SimpleCommunicator, M, T>,
    beam_size: usize,
    layered_registries: Layered<StateRegistry<T, N>>,
    layered_opens: Layered<BinaryHeap<Rc<N>>>,
    layered_beam_remaining: Layered<usize>,
    layered_received_all_remaining: Layered<i32>,
    layered_sent_counters: Layered<Vec<i32>>,
    layered_received_counters: Layered<Vec<i32>>,
    maximum_expanded_depth: usize,
    is_pruned: bool,
    local_dual_bound: Option<T>,
    is_time_out: bool,
    n_remaining_time_out_ack: usize,
    is_checking_termination: bool,
    is_terminated: bool,
}

impl<'a, T, N, M, L, R, B, F, V> Hdbs3<'a, T, N, M, L, R, B, F, V>
where
    T: Numeric + IsFloat + Ord + Display + Hash,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    N: BfsNodeWithDistributedIdChain<T> + From<M> + Clone,
    M: Clone + NodeMessage<T> + From<N>,
    L: FnMut(
        StateInRegistry,
        T,
        &TransitionWithId<V>,
        &DistributedTransitionIdChain,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> NodeGenerationResult<Rc<N>>,
    R: FnMut(
        StateWithHashableSignatureVariables,
        T,
        &TransitionWithId<V>,
        &DistributedTransitionIdChain,
        Option<T>,
    ) -> Option<M>,
    B: FnMut(T, T) -> T,
    F: FnMut(&HashableSignatureVariables) -> u64,
    V: TransitionInterface + Clone + Default + 'static,
    Transition: From<V> + From<TransitionWithId<V>>,
    TransitionWithId<V>: Clone,
{
    const TAG_NODE: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET;
    const TAG_N_SENT: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 1;
    const TAG_TIME_OUT: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 2;
    const TAG_TIME_OUT_ACK: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 3;
    const TAG_TERMINATION_DETECTION: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 4;
    const TAG_TERMINATE: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 5;

    /// Creates a new HDBS3 solver.
    pub fn new(
        input: &SearchInput<'a, M, TransitionWithId<V>>,
        evaluators: MpiAnytimeSearchEvaluators<L, R, B>,
        parameters: MpiAnytimeSearchParameters<T>,
        beam_size: usize,
        hash_function: F,
        communicator: &'a SimpleCommunicator,
    ) -> Self {
        let model = input.generator.model.clone();
        let mut search = MpiAnytimeSearch::new(
            input.generator.clone(),
            input.solution_suffix,
            evaluators,
            parameters,
            hash_function,
            communicator,
        );

        let node_communicator = Hdbs3NodeCommunicator::new(
            communicator,
            Self::TAG_NODE,
            Self::TAG_N_SENT,
            Self::TAG_TERMINATION_DETECTION,
            model.clone(),
        );

        let mut registry = StateRegistry::new(model.clone());
        let mut open = BinaryHeap::default();

        if let Some(node) = search.generate_root_node(input.node.clone(), &mut registry) {
            open.push(node);
        }

        let layered_registries = Layered::new(registry);
        let layered_opens = Layered::new(open);

        let communicator_size = communicator.size();
        let layered_beam_remaining = Layered::new(beam_size);
        let layered_received_all_remaining = Layered::new(0);
        let mut layered_sent_counters = Layered::new(vec![0; communicator_size as usize]);
        layered_sent_counters.pop(0);
        let layered_received_counters = Layered::new(vec![0; communicator_size as usize]);

        Self {
            model,
            search,
            communicator,
            node_communicator,
            beam_size,
            layered_registries,
            layered_opens,
            layered_beam_remaining,
            layered_received_all_remaining,
            layered_sent_counters,
            layered_received_counters,
            maximum_expanded_depth: 0,
            is_pruned: false,
            local_dual_bound: None,
            is_time_out: false,
            n_remaining_time_out_ack: 0,
            is_checking_termination: false,
            is_terminated: false,
        }
    }

    fn open_node(&mut self, node: N, depth: usize) {
        let registry = self
            .layered_registries
            .get_mut_or_create(depth, || StateRegistry::new(self.model.clone()));

        if let Some(node) = self.search.open_node(node, registry) {
            let open = self
                .layered_opens
                .get_mut_or_create(depth, BinaryHeap::default);
            open.push(node.clone());
        }
    }

    fn decrement_received_all_remaining(&mut self, depth: usize) {
        let counter = self
            .layered_received_all_remaining
            .get_mut_or_create(depth, || self.communicator.size());
        *counter -= 1;

        if *counter == 0
            && depth <= self.maximum_expanded_depth
            && self
                .layered_opens
                .get(depth)
                .map_or(true, |open| open.is_empty())
        {
            self.finish_layer(depth);
        }
    }

    fn finish_layer(&mut self, depth: usize) {
        let minimum_depth = self.layered_sent_counters.minimum_depth();

        for d in minimum_depth..=depth + 1 {
            let counters = self
                .layered_sent_counters
                .get_mut_or_create(d, || vec![0; self.communicator.size() as usize]);

            for destination_rank in 0..self.communicator.size() {
                if destination_rank != self.communicator.rank() {
                    let n_sent = counters[destination_rank as usize];
                    self.node_communicator
                        .send_n_sent(destination_rank, n_sent, d);
                }
            }
        }

        self.layered_sent_counters.pop(depth + 1);

        let callback = |open: &mut BinaryHeap<Rc<N>>, _| {
            if let Some(node) = open.peek() {
                if let Some(bound) = node.bound(&self.model) {
                    if !data_structure::exceed_bound(&self.model, bound, self.local_dual_bound) {
                        self.local_dual_bound = Some(bound);
                    }
                }
            }
        };
        self.layered_opens.pop_with(depth, callback);
        self.layered_registries.pop(depth);
        self.layered_beam_remaining.pop(depth);
        self.layered_received_all_remaining.pop(depth);
        self.layered_received_counters.pop(depth);

        self.decrement_received_all_remaining(depth + 1);
    }

    fn update_received_counter(&mut self, depth: usize, rank: Rank, value: i32) {
        let minimum_depth = self.layered_received_all_remaining.minimum_depth();

        if depth >= minimum_depth {
            if depth - 1 > self.maximum_expanded_depth {
                self.maximum_expanded_depth = depth - 1;

                if self.layered_received_all_remaining.get(depth - 1) == Some(&0)
                    && self
                        .layered_opens
                        .get(depth - 1)
                        .map_or(true, |open| open.is_empty())
                {
                    self.finish_layer(depth - 1);
                }
            }

            let counters = self
                .layered_received_counters
                .get_mut_or_create(depth, || vec![0; self.communicator.size() as usize]);
            let rank = rank as usize;
            counters[rank] += value;

            if counters[rank] == 0 {
                self.decrement_received_all_remaining(depth);
            }
        }
    }

    fn receive_node(&mut self, source_rank: Rank) {
        self.search.increment_received();

        let (node, bound, depth) = if self.is_time_out {
            let (bound, depth) = self
                .node_communicator
                .receive_and_discard(source_rank, self.local_dual_bound);

            (None, bound, depth)
        } else {
            let depth_bound = self.layered_opens.minimum_depth();

            self.node_communicator
                .receive(source_rank, self.search.get_primal_bound(), depth_bound)
        };

        if let Some(node) = node {
            let node = N::from(node);
            self.open_node(node, depth);
        } else if let Some(bound) = bound {
            if !data_structure::exceed_bound(&self.model, bound, self.local_dual_bound) {
                self.local_dual_bound = Some(bound);
            }
        }

        self.update_received_counter(depth, source_rank, 1);
    }

    fn receive_n_sent(&mut self, source_rank: Rank) {
        let (n_sent, depth) = self.node_communicator.receive_n_sent(source_rank);
        self.update_received_counter(depth, source_rank, -n_sent);
    }

    fn broadcast_time_out(&mut self) {
        if !self.layered_opens.is_empty() {
            let maximum_depth = self.layered_opens.maximum_depth();
            self.finish_layer(maximum_depth);
        }

        for destination_rank in 0..self.communicator.size() {
            if destination_rank != self.communicator.rank() {
                let buffer: [u8; 0] = [];
                let destination_process = self.communicator.process_at_rank(destination_rank);
                destination_process.buffered_send_with_tag(&buffer, Self::TAG_TIME_OUT);
                self.n_remaining_time_out_ack += 1;
            }
        }
    }

    fn receive_time_out(&mut self, source_rank: Rank) {
        let mut buffer: [u8; 0] = [];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, Self::TAG_TIME_OUT);
        self.is_time_out = true;

        if !self.layered_opens.is_empty() {
            let maximum_depth = self.layered_opens.maximum_depth();
            self.finish_layer(maximum_depth);
        }

        source_process.buffered_send_with_tag(&buffer, Self::TAG_TIME_OUT_ACK);
    }

    fn receive_time_out_ack(&mut self, source_rank: Rank) {
        let mut buffer: [u8; 0] = [];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, Self::TAG_TIME_OUT_ACK);
        self.n_remaining_time_out_ack -= 1;
    }

    fn broadcast_terminate(&mut self) {
        for destination_rank in 0..self.communicator.size() {
            if destination_rank != self.communicator.rank() {
                let buffer: [u8; 0] = [];
                let destination_process = self.communicator.process_at_rank(destination_rank);
                destination_process.buffered_send_with_tag(&buffer, Self::TAG_TERMINATE);
            }
        }

        self.is_terminated = true;
    }

    fn receive_terminate(&mut self, source_rank: Rank) {
        let mut buffer: [u8; 0] = [];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, Self::TAG_TERMINATE);
        self.is_terminated = true;
    }

    fn cannot_terminate(&self) -> bool {
        self.search.cannot_terminate()
            || self.n_remaining_time_out_ack > 0
            || (!self.is_time_out && !self.layered_opens.is_empty())
    }

    fn receive_termination_detection(&mut self, source_rank: Rank) {
        let destination_rank = (self.communicator.rank() + 1) % self.communicator.size();
        let local_invalid = self.cannot_terminate();
        let result = self
            .node_communicator
            .receive_termination_detection_and_forward(
                source_rank,
                destination_rank,
                local_invalid,
            );

        if let Some(result) = result {
            if result {
                self.broadcast_terminate();
            }

            self.is_checking_termination = false;
        }
    }

    fn process_message(&mut self) {
        let any_process = self.communicator.any_process();

        while let Some(status) = any_process.immediate_probe() {
            let source_rank = status.source_rank();
            let tag = status.tag();

            match tag {
                Self::TAG_NODE => self.receive_node(source_rank),
                Self::TAG_N_SENT => self.receive_n_sent(source_rank),
                Self::TAG_TIME_OUT => self.receive_time_out(source_rank),
                Self::TAG_TIME_OUT_ACK => self.receive_time_out_ack(source_rank),
                Self::TAG_TERMINATION_DETECTION => self.receive_termination_detection(source_rank),
                Self::TAG_TERMINATE => self.receive_terminate(source_rank),
                _ => self.search.receive_message(source_rank, tag),
            }
        }
    }

    fn expand(&mut self, keep_buffer: &mut Vec<Rc<N>>, send_buffer: &mut Vec<(Rank, M)>) -> bool {
        let mut maximum_finish_depth = None;
        let mut result = None;

        for (depth, open) in self.layered_opens.iter_mut() {
            let node = open_list::pop_from_open(open, &self.model, self.search.get_primal_bound());

            if open.is_empty() && self.layered_received_all_remaining.get(depth) == Some(&0) {
                maximum_finish_depth = Some(depth);
                self.maximum_expanded_depth = cmp::max(depth, self.maximum_expanded_depth);
            }

            if let Some(node) = node {
                let counter = self
                    .layered_beam_remaining
                    .get_mut_or_create(depth, || self.beam_size);
                *counter -= 1;

                if maximum_finish_depth
                    .map_or(true, |maximum_finish_depth| depth > maximum_finish_depth)
                    && *counter == 0
                {
                    self.is_pruned = true;
                    maximum_finish_depth = Some(depth);
                }

                result = Some((node, depth));
                self.maximum_expanded_depth = cmp::max(depth, self.maximum_expanded_depth);

                break;
            }
        }

        let expanded = result.is_some();

        if let Some((node, depth)) = result {
            let registry = self
                .layered_registries
                .get_mut_or_create(depth + 1, || StateRegistry::new(self.model.clone()));
            self.search.expand(node, registry, keep_buffer, send_buffer);

            for (destination_rank, successor) in send_buffer.drain(..) {
                self.node_communicator
                    .send(destination_rank, &successor, depth + 1);
                let counters = self
                    .layered_sent_counters
                    .get_mut_or_create(depth + 1, || vec![0; self.communicator.size() as usize]);
                counters[destination_rank as usize] += 1;
            }

            for successor in keep_buffer.drain(..) {
                let open = self
                    .layered_opens
                    .get_mut_or_create(depth + 1, BinaryHeap::default);
                open.push(successor);
            }
        }

        if let Some(maximum_finish_depth) = maximum_finish_depth {
            self.finish_layer(maximum_finish_depth);
        }

        expanded
    }

    fn finalize(&mut self) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        let (mut solution, statistics) = self.search.finalize(self.local_dual_bound);

        let root_process = self
            .communicator
            .process_at_rank(self.search.get_root_rank());

        if self.communicator.rank() == self.search.get_root_rank() {
            let mut buffer = [solution.cost.is_some(), solution.best_bound.is_some()];
            root_process.broadcast_into(&mut buffer);

            if solution.cost.is_some() {
                if T::is_float() {
                    let mut buffer = [solution.cost.unwrap().to_continuous()];
                    root_process.broadcast_into(&mut buffer);
                } else {
                    let mut buffer = [solution.cost.unwrap().to_integer()];
                    root_process.broadcast_into(&mut buffer);
                }
            }

            if solution.best_bound.is_some() {
                if T::is_float() {
                    let mut buffer = [solution.best_bound.unwrap().to_continuous()];
                    root_process.broadcast_into(&mut buffer);
                } else {
                    let mut buffer = [solution.best_bound.unwrap().to_integer()];
                    root_process.broadcast_into(&mut buffer);
                }
            }
        } else {
            let mut buffer = [false, false];
            root_process.broadcast_into(&mut buffer);

            if buffer[0] {
                if T::is_float() {
                    let mut buffer = [0.0];
                    root_process.broadcast_into(&mut buffer);
                    solution.cost = Some(T::from_continuous(buffer[0]));
                } else {
                    let mut buffer = [0];
                    root_process.broadcast_into(&mut buffer);
                    solution.cost = Some(T::from_integer(buffer[0]));
                }
            }

            if buffer[1] {
                if T::is_float() {
                    let mut buffer = [0.0];
                    root_process.broadcast_into(&mut buffer);
                    solution.best_bound = Some(T::from_continuous(buffer[0]));
                } else {
                    let mut buffer = [0];
                    root_process.broadcast_into(&mut buffer);
                    solution.best_bound = Some(T::from_integer(buffer[0]));
                }
            }
        }

        if self.is_time_out {
            solution.time_out = true;
            solution.is_optimal = false;
            solution.is_infeasible = false;
        } else {
            let sendbuf = [self.is_pruned];
            let mut recvbuf = vec![false; self.communicator.size() as usize];
            self.communicator
                .all_gather_into(&sendbuf, &mut recvbuf[..]);

            let is_pruned = recvbuf.iter().any(|&x| x);

            if !is_pruned {
                solution.is_optimal = solution.cost.is_some();
                solution.is_infeasible = solution.cost.is_none();

                if solution.is_optimal {
                    solution.best_bound = solution.cost;
                }
            }

            if !solution.is_optimal
                && solution.cost.is_some()
                && solution.cost == solution.best_bound
            {
                solution.is_optimal = true;
            }
        }

        solution.time = self.search.elapsed_time();

        (solution, statistics)
    }

    pub fn search(&mut self) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        let mut keep_buffer = vec![];
        let mut send_buffer = vec![];

        if self.layered_opens.get(0).unwrap().is_empty() {
            self.finish_layer(0);
        }

        loop {
            self.process_message();

            if self.is_terminated {
                break;
            }

            if self.communicator.rank() == self.search.get_root_rank() {
                if !self.is_time_out && self.search.check_time_limit() {
                    self.is_time_out = true;
                    self.broadcast_time_out();
                }

                if self.is_time_out
                    && self.n_remaining_time_out_ack == 0
                    && !self.is_checking_termination
                {
                    self.is_checking_termination = true;
                    let destination_rank =
                        (self.communicator.rank() + 1) % self.communicator.size();
                    self.node_communicator
                        .initiate_termination(destination_rank);
                }
            }

            if self.is_time_out {
                continue;
            }

            if !self.expand(&mut keep_buffer, &mut send_buffer)
                && self.communicator.rank() == self.search.get_root_rank()
                && !self.is_checking_termination
                && !self.cannot_terminate()
            {
                self.is_checking_termination = true;
                let destination_rank = (self.communicator.rank() + 1) % self.communicator.size();
                self.node_communicator
                    .initiate_termination(destination_rank);
            }
        }

        self.communicator.barrier();

        self.finalize()
    }

    pub fn gather_bound_to_expanded(&self) -> Vec<KeyValueStatistics<T, usize>> {
        self.search.gather_bound_to_expanded()
    }
}
