use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure::{
    exceed_bound, HashableSignatureVariables,
};
use dypdl_heuristic_search::search_algorithm::util::TimeKeeper;
use dypdl_heuristic_search::search_algorithm::{SearchInput, Solution, TransitionWithId};
use mpi::traits::*;
use mpi::{topology::SimpleCommunicator, Rank, Tag};
use std::collections::BinaryHeap;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::str::FromStr;

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::is_float::IsFloat;
use crate::mpi_anytime_search::{MpiAnytimeSearch, MpiAnytimeSearchParameters};
use crate::node_communicator::TimeStampedNodeCommunicator;
use crate::node_data_type::NodeDatatype;
use crate::statistics::Statistics;

pub struct HdBestFirstSearch<'a, T, N, M, E, B, F, V = Transition>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    V: TransitionInterface + Clone + Default,
{
    model: Rc<Model>,
    search: MpiAnytimeSearch<'a, T, N, M, E, B, F, V>,
    communicator: &'a SimpleCommunicator,
    node_communicator: TimeStampedNodeCommunicator<'a, SimpleCommunicator, M, T>,
    open: BinaryHeap<Rc<N>>,
    time_keeper: TimeKeeper,
    is_checking_termination: bool,
    is_terminated: bool,
}

impl<'a, T, N, M, E, B, F, V> HdBestFirstSearch<'a, T, N, M, E, B, F, V>
where
    T: Numeric + IsFloat + Ord + Display,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    M: Clone + NodeDatatype<T>,
    E: FnMut(&N, &TransitionWithId<V>, Option<T>) -> Option<M>,
    B: FnMut(T, T) -> T,
    F: FnMut(&HashableSignatureVariables) -> u64,
    V: TransitionInterface + Clone + Default + 'static,
    Transition: From<V> + From<TransitionWithId<V>>,
    TransitionWithId<V>: Clone,
{
    const TAG_NODE: Tag = MpiAnytimeSearch::<'a, T, N, M, E, B, F, V>::TAG_OFFSET;
    const TAG_TERMINATION_DETECTION: Tag =
        MpiAnytimeSearch::<'a, T, N, M, E, B, F, V>::TAG_OFFSET + 1;
    const TAG_TERMINATE: Tag = MpiAnytimeSearch::<'a, T, N, M, E, B, F, V>::TAG_OFFSET + 2;

    /// Creates a new HDACPS solver.
    pub fn new(
        input: SearchInput<'a, M, TransitionWithId<V>>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: MpiAnytimeSearchParameters<T>,
        hash_function: F,
        communicator: &'a SimpleCommunicator,
    ) -> HdBestFirstSearch<'a, T, N, M, E, B, F, V> {
        let mut time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let model = input.generator.model.clone();
        let node_communicator = TimeStampedNodeCommunicator::new(
            communicator,
            Self::TAG_NODE,
            Self::TAG_TERMINATION_DETECTION,
            model.clone(),
        );

        let mut search = MpiAnytimeSearch::new(
            input.generator,
            input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
            parameters,
            hash_function,
            communicator,
        );

        let mut open = BinaryHeap::new();

        search.generate_root_node(input.node, |node| {
            open.push(node);
        });

        time_keeper.stop();

        Self {
            model,
            search,
            communicator,
            node_communicator,
            open,
            time_keeper,
            is_checking_termination: false,
            is_terminated: false,
        }
    }

    fn receive_node(&mut self, source_rank: Rank) {
        self.search.increment_received();

        if let Some(node) = self
            .node_communicator
            .receive(source_rank, self.search.get_primal_bound())
        {
            let callback = |node| {
                self.open.push(node);
            };

            let node = N::from(node);
            self.search.open_node(node, callback);
        }
    }

    fn broadcast_terminate(&mut self) {
        for destination_rank in 0..self.communicator.size() {
            if destination_rank != self.communicator.rank() {
                let buffer: [u8; 0] = [];
                self.communicator
                    .process_at_rank(destination_rank)
                    .buffered_send_with_tag(&buffer, Self::TAG_TERMINATE);
            }
        }

        self.is_terminated = true;
    }

    fn receive_terminate(&mut self, source_rank: Rank) {
        let mut buffer: [u8; 0] = [];
        self.communicator
            .process_at_rank(source_rank)
            .receive_into_with_tag(&mut buffer, Self::TAG_TERMINATE);
        self.is_terminated = true;
    }

    fn receive_termination_detection(&mut self, source_rank: Rank) {
        let destination_rank = (self.communicator.rank() + 1) % self.communicator.size();
        let local_invalid = self.search.cannot_terminate() || !self.open.is_empty();
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
                Self::TAG_TERMINATION_DETECTION => self.receive_termination_detection(source_rank),
                Self::TAG_TERMINATE => self.receive_terminate(source_rank),
                _ => self.search.receive_message(source_rank, tag),
            }
        }
    }

    pub fn search(mut self) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        self.time_keeper.start();

        let mut time_out = false;

        loop {
            self.process_message();

            if self.is_terminated {
                break;
            }

            if self.communicator.rank() == self.search.get_root_rank()
                && self.time_keeper.check_time_limit(self.search.is_quiet())
            {
                time_out = true;
                self.broadcast_terminate();
                self.is_terminated = true;
                break;
            }

            if let Some(node) = self.open.pop() {
                if node.is_closed() {
                    continue;
                }
                node.close();

                if let Some(dual_bound) = node.bound(&self.model) {
                    if exceed_bound(&self.model, dual_bound, self.search.get_primal_bound()) {
                        if N::ordered_by_bound() {
                            self.open.clear();
                        }
                        continue;
                    }
                }

                let local_callback = |successor| {
                    self.open.push(successor);
                };

                let send_callback = |destination_rank, successor| {
                    self.node_communicator.send(destination_rank, &successor);
                };

                self.search.expand(node, local_callback, send_callback);
            } else if !self.is_checking_termination {
                self.is_checking_termination = true;
                let destination_rank = (self.communicator.rank() + 1) % self.communicator.size();
                self.node_communicator
                    .initiate_termination(destination_rank);
            }
        }

        let (mut solution, statistics) = self.search.finalize(None);

        if time_out {
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

        solution.time = self.time_keeper.elapsed_time();

        (solution, statistics)
    }
}
