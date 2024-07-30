use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::{HashableSignatureVariables, StateWithHashableSignatureVariables},
    SearchInput, Solution, StateInRegistry, StateRegistry, TransitionWithId,
};
use mpi::{topology::SimpleCommunicator, traits::*, Rank, Tag};
use std::hash::Hash;
use std::rc::Rc;
use std::str::FromStr;
use std::{
    fmt::{Debug, Display},
    vec,
};

use crate::layered_beams::{LayeredBeams, LayeredPerRankCounters};
use crate::node_communicator::TimeStampedNodeDepthCommunicator;
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
    layered_beams: LayeredBeams<T, N>,
    rank_to_n_sent: LayeredPerRankCounters,
    rank_to_counters: LayeredPerRankCounters,
    depth_to_received_all: Vec<i32>,
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
    const TAG_N_INITIAL_CLOSED_NODES: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET;
    const TAG_N_INITIAL_OPEN_NODES: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 1;
    const TAG_NODE: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 2;
    const TAG_ALL_NODES_SENT_IN_LAYER: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 3;
    const TAG_TIME_OUT: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 4;
    const TAG_TIME_OUT_ACK: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 5;
    const TAG_TERMINATION_DETECTION: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 6;
    const TAG_TERMINATE: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 7;

    /// Creates a new HDBS3 solver.
    pub fn new(
        input: SearchInput<'a, M, TransitionWithId<V>>,
        evaluators: MpiAnytimeSearchEvaluators<L, R, B>,
        parameters: MpiAnytimeSearchParameters<T>,
        beam_size: usize,
        hash_function: F,
        communicator: &'a SimpleCommunicator,
    ) -> Self {
        let model = input.generator.model.clone();
        let mut search = MpiAnytimeSearch::new(
            input.generator,
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

        let mut layered_beams = LayeredBeams::new(model.clone(), beam_size);
        let node_generator = |registry: &mut _| search.generate_root_node(input.node, registry);

        layered_beams.insert_with(node_generator, 0);

        let rank_to_n_sent = LayeredPerRankCounters::new(communicator.size() as usize);
        let rank_to_n_received = LayeredPerRankCounters::new(communicator.size() as usize);
        let depth_to_received_all = vec![communicator.size()];

        Self {
            model,
            search,
            communicator,
            node_communicator,
            beam_size,
            layered_beams,
            rank_to_n_sent,
            rank_to_counters: rank_to_n_received,
            depth_to_received_all,
            local_dual_bound: None,
            is_time_out: false,
            n_remaining_time_out_ack: 0,
            is_checking_termination: false,
            is_terminated: false,
        }
    }

    fn open_node(&mut self, node: N, depth: usize) {
        let node_generator = |registry: &mut _| self.search.open_node(node, registry);
        self.layered_beams.insert_with(node_generator, depth);
    }

    fn increment_received_all(&mut self, depth: usize) {
        while self.depth_to_received_all.len() < depth + 1 {
            self.depth_to_received_all.push(0);
        }

        self.depth_to_received_all[depth] += 1;
    }

    fn check_count(&mut self, source_rank: i32, depth: usize) {
        if self.rank_to_counters.get_count(source_rank as usize, depth) == 0 {
            self.increment_received_all(depth);

            if self.depth_to_received_all[depth] == self.communicator.size() {
                self.layered_beams.notify_generated_all(depth);
            }
        }
    }

    fn receive_node(&mut self, source_rank: Rank) {
        self.search.increment_received();

        if self.is_time_out {
            if let Some(bound) = self
                .node_communicator
                .receive_and_discard(source_rank, self.local_dual_bound)
            {
                if N::ordered_by_bound() {
                    self.local_dual_bound = Some(bound);
                }
            }
        } else if let Some((node, depth)) = self
            .node_communicator
            .receive(source_rank, self.search.get_primal_bound())
        {
            let node = N::from(node);
            self.open_node(node, depth);
        }

        let depth = self.node_communicator.get_depth();
        self.rank_to_counters.increment(source_rank as usize, depth);
        self.check_count(source_rank, depth);
    }

    fn recieve_sent_all_nodes_in_layer(&mut self, source_rank: Rank) {
        let mut buffer = [0; 2];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, Self::TAG_ALL_NODES_SENT_IN_LAYER);
        let depth = buffer[0] as usize;
        let n_sent = buffer[1];
        self.rank_to_counters
            .decrease(source_rank as usize, depth, n_sent);
        self.check_count(source_rank, depth);
    }

    fn broadcast_time_out(&mut self) {
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
        self.local_dual_bound = self.compute_local_dual_bound();
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
            || (!self.is_time_out && self.layered_beams.cannot_terminate())
    }

    fn receive_termination_detection(&mut self, source_rank: Rank) {
        let destination_rank = (self.communicator.rank() + 1) % self.communicator.size();
        let local_invalid = self.search.cannot_terminate()
            || self.n_remaining_time_out_ack > 0
            || (!self.is_time_out && !self.layered_beams.cannot_terminate());
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
                    self.recieve_sent_all_nodes_in_layer(source_rank)
                }
                Self::TAG_TIME_OUT => self.receive_time_out(source_rank),
                Self::TAG_TIME_OUT_ACK => self.receive_time_out_ack(source_rank),
                Self::TAG_TERMINATION_DETECTION => self.receive_termination_detection(source_rank),
                Self::TAG_TERMINATE => self.receive_terminate(source_rank),
                _ => self.search.receive_message(source_rank, tag),
            }
        }
    }

    fn pop_node_and_depth(&mut self) -> Option<(Rc<N>, usize)> {
        None
    }

    fn compute_local_dual_bound(&self) -> Option<T> {
        if N::ordered_by_bound() {
            self.layered_beams.get_local_dual_bound()
        } else {
            None
        }
    }

    pub fn search(&mut self) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        let mut keep_buffer = vec![];
        let mut send_buffer = vec![];

        loop {
            self.process_message();

            if self.is_terminated {
                break;
            }

            if self.communicator.rank() == self.search.get_root_rank() {
                if !self.is_time_out && self.search.check_time_limit() {
                    self.is_time_out = true;
                    self.local_dual_bound = self.compute_local_dual_bound();
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

            if let Some(result) = self.layered_beams.pop(self.search.get_primal_bound()) {
                let node = result.node;
                let depth = result.depth;
                let registry = self.layered_beams.get_registry_mut(depth + 1);
                self.search
                    .expand(node, registry, &mut keep_buffer, &mut send_buffer);

                for (destination_rank, successor) in send_buffer.drain(..) {
                    self.node_communicator
                        .send(destination_rank, &successor, depth + 1);
                    self.rank_to_n_sent
                        .increment(destination_rank as usize, depth + 1);
                }

                if result.is_last {
                    self.increment_received_all(depth + 1);

                    while self.layered_beams.current_minimum_depth() <= depth + 1 {
                        let d = self.layered_beams.current_minimum_depth();

                        for destination_rank in 0..self.communicator.size() {
                            if destination_rank != self.communicator.rank() {
                                let destination =
                                    self.communicator.process_at_rank(destination_rank);
                                let n_sent =
                                    self.rank_to_n_sent.get_count(destination_rank as usize, d);
                                let buffer = [d as i32, n_sent];
                                destination.buffered_send_with_tag(
                                    &buffer,
                                    Self::TAG_ALL_NODES_SENT_IN_LAYER,
                                );
                            }
                        }

                        self.rank_to_n_sent.clear_depth(d);

                        if d < depth {
                            self.layered_beams.clear_minimum_depth();
                        }
                    }
                }

                for successor in keep_buffer.drain(..) {
                    self.layered_beams.insert(successor, depth + 1);
                }
            } else if self.communicator.rank() == self.search.get_root_rank()
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

        let (mut solution, statistics) = self.search.finalize(self.local_dual_bound);

        if self.is_time_out {
            solution.time_out = true;
            solution.is_optimal = false;
            solution.is_infeasible = false;
        } else {
            solution.is_optimal = solution.cost.is_some();
            solution.is_infeasible = solution.cost.is_none();

            if solution.is_optimal {
                solution.best_bound = solution.cost;
            }
        }

        if !solution.is_optimal && solution.cost.is_some() && solution.cost == solution.best_bound {
            solution.is_optimal = true;
        }

        solution.time = self.search.elapsed_time();

        (solution, statistics)
    }

    pub fn gather_bound_to_expanded(&self) -> Vec<KeyValueStatistics<T, usize>> {
        self.search.gather_bound_to_expanded()
    }
}
