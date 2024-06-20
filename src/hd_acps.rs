use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    search_algorithm::{
        data_structure::{self, HashableSignatureVariables, StateWithHashableSignatureVariables},
        SearchInput, Solution, StateInRegistry, StateRegistry, TransitionWithId,
    },
    ProgressiveSearchParameters,
};
use mpi::{topology::SimpleCommunicator, traits::*, Rank, Tag};
use std::collections::BinaryHeap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::rc::Rc;
use std::str::FromStr;

use crate::is_float::IsFloat;
use crate::key_value_statistics::KeyValueStatistics;
use crate::mpi_anytime_search::{
    MpiAnytimeSearch, MpiAnytimeSearchEvaluators, MpiAnytimeSearchParameters,
};
use crate::node_communicator::TimeStampedNodeDepthCommunicator;
use crate::node_message::NodeMessage;
use crate::statistics::Statistics;
use crate::{
    bfs_node_with_distributed_id_chain::{BfsNodeWithDistributedIdChain, NodeGenerationResult},
    distributed_id_chain::DistributedTransitionIdChain,
};

pub struct HdAcps<'a, T, N, M, L, R, B, F, V = Transition>
where
    T: IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    V: TransitionInterface + Clone + Default,
{
    model: Rc<Model>,
    search: MpiAnytimeSearch<'a, T, N, M, L, R, B, F, V>,
    communicator: &'a SimpleCommunicator,
    node_communicator: TimeStampedNodeDepthCommunicator<'a, SimpleCommunicator, M, T>,
    progressive_parameters: ProgressiveSearchParameters,
    width: usize,
    open: Vec<BinaryHeap<Rc<N>>>,
    local_dual_bound: Option<T>,
    is_time_out: bool,
    n_remaining_time_out_ack: usize,
    is_checking_termination: bool,
    is_terminated: bool,
}

impl<'a, T, N, M, L, R, B, F, V> HdAcps<'a, T, N, M, L, R, B, F, V>
where
    T: IsFloat + Ord + Display + Hash,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    M: Clone + NodeMessage<T>,
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
    const TAG_TIME_OUT: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 1;
    const TAG_TIME_OUT_ACK: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 2;
    const TAG_TERMINATION_DETECTION: Tag =
        MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 3;
    const TAG_TERMINATE: Tag = MpiAnytimeSearch::<'a, T, N, M, L, R, B, F, V>::TAG_OFFSET + 4;

    /// Creates a new HDACPS solver.
    pub fn new(
        input: SearchInput<'a, M, TransitionWithId<V>>,
        evaluators: MpiAnytimeSearchEvaluators<L, R, B>,
        parameters: MpiAnytimeSearchParameters<T>,
        progressive_parameters: ProgressiveSearchParameters,
        hash_function: F,
        communicator: &'a SimpleCommunicator,
    ) -> HdAcps<'a, T, N, M, L, R, B, F, V> {
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

        let mut open = vec![BinaryHeap::new()];

        if let Some(node) = search.generate_root_node(input.node) {
            open[0].push(node);
        }

        Self {
            model,
            search,
            communicator,
            node_communicator,
            progressive_parameters,
            width: progressive_parameters.init,
            open,
            local_dual_bound: None,
            is_time_out: false,
            n_remaining_time_out_ack: 0,
            is_checking_termination: false,
            is_terminated: false,
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

            if let Some(node) = self.search.open_node(node) {
                while depth >= self.open.len() {
                    self.open.push(BinaryHeap::new());
                }

                self.open[depth].push(node);
            }
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
            || (!self.is_time_out && self.open.iter().any(|o| !o.is_empty()));
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

    fn compute_local_dual_bound(&self) -> Option<T> {
        if N::ordered_by_bound() {
            let dual_bound_iter = self
                .open
                .iter()
                .filter_map(|o| o.peek().and_then(|n| n.bound(&self.model)));

            if self.model.reduce_function == ReduceFunction::Max {
                dual_bound_iter.max()
            } else {
                dual_bound_iter.min()
            }
        } else {
            None
        }
    }

    pub fn search(&mut self) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        let mut current_depth = 0;
        let mut no_node = true;
        let mut goal_found = false;
        let mut keep_buffer = vec![];
        let mut send_buffer = vec![];

        'outer: loop {
            let mut popped = 0;

            while popped < self.width {
                self.process_message();

                if self.is_time_out || self.is_terminated {
                    break 'outer;
                }

                if self.communicator.rank() == self.search.get_root_rank()
                    && self.search.check_time_limit()
                {
                    self.is_time_out = true;
                    self.local_dual_bound = self.compute_local_dual_bound();
                    self.broadcast_time_out();
                    break 'outer;
                }

                if self.open[current_depth].is_empty() {
                    break;
                }

                let node = self.open[current_depth].pop().unwrap();

                if node.is_closed() {
                    continue;
                }
                node.close();

                if let Some(dual_bound) = node.bound(&self.model) {
                    if data_structure::exceed_bound(
                        &self.model,
                        dual_bound,
                        self.search.get_primal_bound(),
                    ) {
                        if N::ordered_by_bound() {
                            self.open[current_depth].clear();
                        }
                        continue;
                    }
                }

                if no_node {
                    no_node = false;
                }

                popped += 1;
                goal_found |= self.search.expand(node, &mut keep_buffer, &mut send_buffer);

                for (destination_rank, successor) in send_buffer.drain(..) {
                    self.node_communicator
                        .send(destination_rank, &successor, current_depth + 1);
                }

                for successor in keep_buffer.drain(..) {
                    while current_depth + 1 >= self.open.len() {
                        self.open.push(BinaryHeap::default());
                    }

                    self.open[current_depth + 1].push(successor);
                }
            }

            if goal_found {
                current_depth = 0;
                no_node = true;
                goal_found = false;

                if self.progressive_parameters.reset {
                    self.width = self.progressive_parameters.init;
                } else {
                    self.width = self.progressive_parameters.increase_width(self.width);
                }
            } else if current_depth + 1 == self.open.len() {
                if self.communicator.rank() == self.search.get_root_rank()
                    && no_node
                    && !self.is_checking_termination
                    && !self.search.cannot_terminate()
                    && self.n_remaining_time_out_ack == 0
                {
                    self.is_checking_termination = true;
                    let destination_rank =
                        (self.communicator.rank() + 1) % self.communicator.size();
                    self.node_communicator
                        .initiate_termination(destination_rank);
                }

                current_depth = 0;
                no_node = true;

                if self.open.iter().any(|o| !o.is_empty()) {
                    self.width = self.progressive_parameters.increase_width(self.width);
                }
            } else {
                current_depth += 1;
            }
        }

        if self.is_time_out {
            loop {
                self.process_message();

                if self.is_terminated {
                    break;
                }

                if self.communicator.rank() == self.search.get_root_rank()
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
