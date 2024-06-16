use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::{self, HashableSignatureVariables, StateWithHashableSignatureVariables},
    SearchInput, Solution, StateInRegistry, StateRegistry, TransitionWithId,
};
use mpi::{topology::SimpleCommunicator, traits::*, Rank, Tag};
use std::collections::BinaryHeap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::rc::Rc;
use std::str::FromStr;

use crate::statistics::Statistics;
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
use crate::{initiation, node_data_type::NodeDatatype};
use crate::{node_communicator::TimeStampedNodeDepthCommunicator, InitiationResult};

pub struct HdHac<'a, T, N, M, L, R, B, F, V = Transition>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    V: TransitionInterface + Clone + Default,
{
    model: Rc<Model>,
    search: MpiAnytimeSearch<'a, T, N, M, L, R, B, F, V>,
    communicator: &'a SimpleCommunicator,
    node_communicator: TimeStampedNodeDepthCommunicator<'a, SimpleCommunicator, M, T>,
    open: BinaryHeap<(Rc<N>, usize)>,
    layered_open: Vec<BinaryHeap<Rc<N>>>,
    current_depth: usize,
    is_layered_turn: bool,
    local_dual_bound: Option<T>,
    is_time_out: bool,
    n_remaining_time_out_ack: usize,
    is_checking_termination: bool,
    is_terminated: bool,
}

impl<'a, T, N, M, L, R, B, F, V> HdHac<'a, T, N, M, L, R, B, F, V>
where
    T: Numeric + IsFloat + Ord + Display + Hash,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    N: BfsNodeWithDistributedIdChain<T> + From<M> + Clone,
    M: Clone + NodeDatatype<T> + From<N>,
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
    const TAG_TIME_OUT: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 3;
    const TAG_TIME_OUT_ACK: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 4;
    const TAG_TERMINATION_DETECTION: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 5;
    const TAG_TERMINATE: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 6;

    /// Creates a new HDACPS solver.
    pub fn new(
        input: SearchInput<'a, M, TransitionWithId<V>>,
        evaluators: MpiAnytimeSearchEvaluators<L, R, B>,
        parameters: MpiAnytimeSearchParameters<T>,
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

        let mut open = BinaryHeap::new();
        let layered_open = vec![BinaryHeap::new()];

        if let Some(node) = search.generate_root_node(input.node) {
            open.push((node, 0));
        }

        Self {
            model,
            search,
            communicator,
            node_communicator,
            open,
            layered_open,
            current_depth: 0,
            is_layered_turn: false,
            local_dual_bound: None,
            is_time_out: false,
            n_remaining_time_out_ack: 0,
            is_checking_termination: false,
            is_terminated: false,
        }
    }

    pub fn distriute_initial_nodes(
        &mut self,
        initiation_result: InitiationResult<T, N, V>,
        ranks: &[Rank],
    ) {
        self.search.initiate(&initiation_result);

        let mut rank_to_open_buffer = vec![Vec::default(); self.communicator.size() as usize];
        let mut rank_to_closed_buffer = vec![Vec::default(); self.communicator.size() as usize];

        for ((_, v), rank) in initiation_result.nodes.into_iter().zip(ranks) {
            if *rank == self.communicator.rank() {
                for node in v {
                    let depth =
                        initiation::get_depth(node.as_ref(), &initiation_result.id_to_depth);

                    if depth == 0 {
                        continue;
                    }

                    if node.is_closed() {
                        self.search.close_node((*node).clone());
                    } else {
                        self.open_node((*node).clone(), depth);
                    }
                }
            } else {
                for node in v {
                    let depth =
                        initiation::get_depth(node.as_ref(), &initiation_result.id_to_depth);

                    if depth == 0 {
                        continue;
                    }

                    let message = M::from((*node).clone());
                    message.set_parent_rank(self.communicator.rank());

                    if node.is_closed() {
                        rank_to_closed_buffer[*rank as usize].push((message, depth));
                    } else {
                        rank_to_open_buffer[*rank as usize].push((message, depth));
                    }
                }
            }
        }

        for (rank, buffer) in rank_to_closed_buffer.into_iter().enumerate() {
            let rank = rank as Rank;

            if rank != self.communicator.rank() {
                let destination_process = self.communicator.process_at_rank(rank);
                let n_nodes = buffer.len();
                destination_process.send_with_tag(&n_nodes, Self::TAG_N_INITIAL_CLOSED_NODES);

                for (message, depth) in buffer {
                    self.node_communicator.send(rank, &message, depth);
                }
            }
        }

        for (rank, buffer) in rank_to_open_buffer.into_iter().enumerate() {
            let rank = rank as Rank;

            if rank != self.communicator.rank() {
                let destination_process = self.communicator.process_at_rank(rank);
                let n_nodes = buffer.len();
                destination_process.send_with_tag(&n_nodes, Self::TAG_N_INITIAL_OPEN_NODES);

                for (message, depth) in buffer {
                    self.node_communicator.send(rank, &message, depth);
                }
            }
        }
    }

    pub fn receive_initial_nodes(&mut self, source_rank: Rank) {
        let source = self.communicator.process_at_rank(source_rank);
        let mut n_closed_nodes = 0;
        source
            .receive_into_with_tag::<usize>(&mut n_closed_nodes, Self::TAG_N_INITIAL_CLOSED_NODES);

        for _ in 0..n_closed_nodes {
            self.search.increment_received();

            if let Some((node, _)) = self
                .node_communicator
                .receive(source_rank, self.search.get_primal_bound())
            {
                let node = N::from(node);
                self.search.close_node(node);
            }
        }

        let mut n_open_nodes = 0;
        source.receive_into_with_tag::<usize>(&mut n_open_nodes, Self::TAG_N_INITIAL_OPEN_NODES);

        for _ in 0..n_open_nodes {
            self.receive_node(source_rank);
        }
    }

    pub fn close_root_node(&mut self, node: M) {
        self.search.close_root_node(node)
    }

    fn open_node(&mut self, node: N, depth: usize) {
        if let Some(node) = self.search.open_node(node) {
            self.open.push((node.clone(), depth));

            while depth >= self.layered_open.len() {
                self.layered_open.push(BinaryHeap::new());
            }

            self.layered_open[depth].push(node);
        }
    }

    fn receive_node(&mut self, source_rank: Rank) {
        self.search.increment_received();

        if self.is_time_out {
            if let Some(bound) = self
                .node_communicator
                .receive_and_discard(source_rank, self.local_dual_bound)
            {
                self.local_dual_bound = Some(bound);
            }
        } else if let Some((node, depth)) = self
            .node_communicator
            .receive(source_rank, self.search.get_primal_bound())
        {
            let node = N::from(node);
            self.open_node(node, depth);
        }
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

    fn receive_termination_detection(&mut self, source_rank: Rank) {
        let destination_rank = (self.communicator.rank() + 1) % self.communicator.size();
        let local_invalid = self.search.cannot_terminate()
            || self.n_remaining_time_out_ack > 0
            || (!self.is_time_out && !self.open.is_empty());
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
                Self::TAG_TIME_OUT => self.receive_time_out(source_rank),
                Self::TAG_TIME_OUT_ACK => self.receive_time_out_ack(source_rank),
                Self::TAG_TERMINATION_DETECTION => self.receive_termination_detection(source_rank),
                Self::TAG_TERMINATE => self.receive_terminate(source_rank),
                _ => self.search.receive_message(source_rank, tag),
            }
        }
    }

    fn pop_from_open(&mut self) -> Option<(Rc<N>, usize)> {
        while let Some((node, depth)) = self.open.pop() {
            if node.is_closed() {
                continue;
            }

            node.close();

            if node.bound(&self.model).map_or(false, |dual_bound| {
                data_structure::exceed_bound(
                    &self.model,
                    dual_bound,
                    self.search.get_primal_bound(),
                )
            }) {
                if N::ordered_by_bound() {
                    self.open.clear();
                }
            } else {
                return Some((node, depth));
            }
        }

        None
    }

    fn pop_from_layered_open(&mut self) -> Option<(Rc<N>, usize)> {
        while let Some(node) = self.layered_open[self.current_depth].pop() {
            if node.is_closed() {
                continue;
            }

            node.close();

            if node.bound(&self.model).map_or(false, |dual_bound| {
                data_structure::exceed_bound(
                    &self.model,
                    dual_bound,
                    self.search.get_primal_bound(),
                )
            }) {
                if N::ordered_by_bound() {
                    self.layered_open[self.current_depth].clear();
                }
            } else {
                return Some((node, self.current_depth));
            }
        }

        None
    }

    fn pop_node_and_depth(&mut self) -> Option<(Rc<N>, usize)> {
        if self.is_layered_turn {
            if self.current_depth > self.layered_open.len() - 1 {
                self.current_depth = 0;
            }

            let initial_depth = self.current_depth;

            loop {
                let result = self.pop_from_layered_open();
                self.current_depth += 1;

                if result.is_some() {
                    self.is_layered_turn = false;

                    return result;
                } else {
                    if self.current_depth > self.layered_open.len() - 1 {
                        self.current_depth = 0;
                    }

                    if self.current_depth == initial_depth {
                        break;
                    }
                }
            }
        }

        self.is_layered_turn = true;

        self.pop_from_open()
    }

    fn compute_local_dual_bound(&self) -> Option<T> {
        self.open
            .peek()
            .and_then(|(node, _)| node.bound(&self.model))
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

            if let Some((node, depth)) = self.pop_node_and_depth() {
                self.search.expand(node, &mut keep_buffer, &mut send_buffer);

                for (destination_rank, successor) in send_buffer.drain(..) {
                    self.node_communicator
                        .send(destination_rank, &successor, depth + 1);
                }

                for successor in keep_buffer.drain(..) {
                    self.open.push((successor.clone(), depth + 1));

                    while depth + 1 >= self.layered_open.len() {
                        self.layered_open.push(BinaryHeap::new());
                    }

                    self.layered_open[depth + 1].push(successor);
                }
            } else if self.communicator.rank() == self.search.get_root_rank()
                && !self.is_checking_termination
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
