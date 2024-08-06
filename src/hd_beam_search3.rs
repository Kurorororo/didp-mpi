use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::{self, HashableSignatureVariables, StateWithHashableSignatureVariables},
    SearchInput, Solution, StateInRegistry, StateRegistry, TransitionWithId,
};
use mpi::{topology::SimpleCommunicator, traits::*, Rank, Tag};
use std::cmp;
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::rc::Rc;
use std::str::FromStr;
use std::{
    fmt::{Debug, Display},
    vec,
};

use crate::layered::Layered;
use crate::node_communicator::TimeStampedNodeDepthCommunicator;
use crate::open_list;
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
use crate::{node_message::NodeMessage, statistics::Statistics};

pub struct Hdbs3<'a, T, N, M, L, R, B, F, V = Transition>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    V: TransitionInterface + Clone + Default,
{
    model: Rc<Model>,
    search: MpiAnytimeSearch<'a, T, N, M, L, R, B, F, V>,
    communicator: &'a SimpleCommunicator,
    node_communicator: TimeStampedNodeDepthCommunicator<'a, SimpleCommunicator, M, T>,
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
    n_remaining_finished_all: usize,
    n_remaining_finished_all_ack: usize,
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
    const TAG_ALL_NODES_SENT_IN_LAYER: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 1;
    const TAG_ALL_NODES_SENT_IN_LAYER_ACK: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 2;
    const TAG_TIME_OUT: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 3;
    const TAG_TIME_OUT_ACK: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 4;
    const TAG_TERMINATION_DETECTION: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 5;
    const TAG_TERMINATE: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 6;

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

        let node_communicator = TimeStampedNodeDepthCommunicator::new(
            communicator,
            Self::TAG_NODE,
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
            n_remaining_finished_all: (communicator_size - 1) as usize,
            n_remaining_finished_all_ack: 0,
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

    fn finish_layer(&mut self, depth: usize) {
        let minimum_depth = self.layered_sent_counters.minimum_depth();

        for d in minimum_depth..=depth + 1 {
            let counters = self
                .layered_sent_counters
                .get_mut_or_create(d, || vec![0; self.communicator.size() as usize]);

            for destination_rank in 0..self.communicator.size() {
                if destination_rank != self.communicator.rank() {
                    let destination = self.communicator.process_at_rank(destination_rank);
                    let n_sent = counters[destination_rank as usize];
                    let buffer = [d as i32, n_sent];
                    destination.buffered_send_with_tag(&buffer, Self::TAG_ALL_NODES_SENT_IN_LAYER);
                }
            }
        }

        self.layered_sent_counters.pop(depth + 1);
        self.maximum_expanded_depth = cmp::max(self.maximum_expanded_depth, depth);

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

        let counter = self
            .layered_received_all_remaining
            .get_mut_or_create(depth + 1, || self.communicator.size());
        *counter -= 1;

        if *counter == 0
            && depth < self.maximum_expanded_depth
            && self
                .layered_opens
                .get(depth + 1)
                .map_or(true, |open| open.is_empty())
        {
            self.finish_layer(depth + 1);
        }
    }

    fn update_received_counter(&mut self, depth: usize, rank: Rank, value: i32) {
        let minimum_depth = self.layered_received_all_remaining.minimum_depth();

        if depth >= minimum_depth {
            if depth - 1 > self.maximum_expanded_depth
                && self.layered_received_all_remaining.get(depth - 1) == Some(&0)
            {
                self.finish_layer(depth - 1);
            }

            self.maximum_expanded_depth = cmp::max(self.maximum_expanded_depth, depth - 1);
            let counters = self
                .layered_received_counters
                .get_mut_or_create(depth, || vec![0; self.communicator.size() as usize]);
            let rank = rank as usize;
            counters[rank] += value;

            if counters[rank] == 0 {
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
        }
    }

    fn receive_node(&mut self, source_rank: Rank) {
        self.search.increment_received();

        let depth = if self.is_time_out {
            let (bound, depth) = self
                .node_communicator
                .receive_depth_and_discard(source_rank, self.local_dual_bound);

            if let Some(bound) = bound {
                if N::ordered_by_bound() {
                    self.local_dual_bound = Some(bound);
                }
            }

            depth
        } else {
            let depth_bound = self.layered_opens.minimum_depth();
            let (node, depth) = self.node_communicator.receive_with_depth_bound(
                source_rank,
                self.search.get_primal_bound(),
                depth_bound,
            );

            if let Some(node) = node {
                let node = N::from(node);
                self.open_node(node, depth);
            }

            depth
        };

        self.update_received_counter(depth, source_rank, 1);
    }

    fn receive_all_nodes_sent_in_layer(&mut self, source_rank: Rank) {
        let mut buffer = [0; 2];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, Self::TAG_ALL_NODES_SENT_IN_LAYER);
        let depth = buffer[0];
        let n_sent = buffer[1];

        if depth < 0 {
            self.n_remaining_finished_all -= 1;

            if self.n_remaining_finished_all == 0 {
                let buffer: [u8; 0] = [];
                let root = self
                    .communicator
                    .process_at_rank(self.search.get_root_rank());
                root.buffered_send_with_tag(&buffer, Self::TAG_ALL_NODES_SENT_IN_LAYER_ACK);
            }
        } else {
            self.update_received_counter(depth as usize, source_rank, -n_sent);
        }
    }

    fn receive_all_nodes_sent_in_layer_ack(&mut self, source_rank: Rank) {
        let mut buffer: [u8; 0] = [];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, Self::TAG_ALL_NODES_SENT_IN_LAYER_ACK);
        self.n_remaining_finished_all_ack -= 1;
    }

    fn broadcast_time_out(&mut self) {
        for destination_rank in 0..self.communicator.size() {
            if destination_rank != self.communicator.rank() {
                let buffer: [u8; 0] = [];
                let destination_process = self.communicator.process_at_rank(destination_rank);
                destination_process.buffered_send_with_tag(&buffer, Self::TAG_TIME_OUT);
                self.n_remaining_time_out_ack += 1;
                self.n_remaining_finished_all_ack += 1;
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

        for destination_rank in 0..self.communicator.size() {
            if destination_rank != self.communicator.rank() {
                let buffer = [-1, -1];
                let destination_process = self.communicator.process_at_rank(destination_rank);
                destination_process
                    .buffered_send_with_tag(&buffer, Self::TAG_ALL_NODES_SENT_IN_LAYER);
            }
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
            || (!self.is_time_out
                && (self
                    .layered_received_all_remaining
                    .get(self.maximum_expanded_depth + 1)
                    != Some(&0)
                    || !self.layered_opens.is_empty()))
            || (self.is_time_out
                && (self.n_remaining_finished_all > 0 || self.n_remaining_finished_all_ack > 0))
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
                Self::TAG_ALL_NODES_SENT_IN_LAYER => {
                    self.receive_all_nodes_sent_in_layer(source_rank)
                }
                Self::TAG_ALL_NODES_SENT_IN_LAYER_ACK => {
                    self.receive_all_nodes_sent_in_layer_ack(source_rank)
                }
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

            self.maximum_expanded_depth = cmp::max(depth, self.maximum_expanded_depth);
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
