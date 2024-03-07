use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure::{
    exceed_bound, HashableSignatureVariables,
};
use dypdl_heuristic_search::search_algorithm::{SearchInput, Solution, TransitionWithId};
use dypdl_heuristic_search::ProgressiveSearchParameters;
use mpi::traits::*;
use mpi::{topology::SimpleCommunicator, Rank, Tag};
use std::cmp;
use std::collections::BinaryHeap;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::str::FromStr;

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::is_float::IsFloat;
use crate::mpi_anytime_search::{MpiAnytimeSearch, MpiAnytimeSearchParameters};
use crate::node_communicator::TimeStampedNodeDepthCommunicator;
use crate::node_data_type::NodeDatatype;
use crate::statistics::Statistics;

pub struct HdApps<'a, T, N, M, E, B, F, V = Transition>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    V: TransitionInterface + Clone + Default,
{
    model: Rc<Model>,
    search: MpiAnytimeSearch<'a, T, N, M, E, B, F, V>,
    communicator: &'a SimpleCommunicator,
    node_communicator: TimeStampedNodeDepthCommunicator<'a, SimpleCommunicator, M, T>,
    progressive_parameters: ProgressiveSearchParameters,
    width: usize,
    open: BinaryHeap<(Rc<N>, usize)>,
    children: BinaryHeap<(Rc<N>, usize)>,
    suspend: BinaryHeap<(Rc<N>, usize)>,
    current_depth: usize,
    is_time_out: bool,
    n_remaining_time_out_ack: usize,
    is_checking_termination: bool,
    is_terminated: bool,
}

impl<'a, T, N, M, E, B, F, V> HdApps<'a, T, N, M, E, B, F, V>
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
    const TAG_TIME_OUT: Tag = MpiAnytimeSearch::<'a, T, N, M, E, B, F, V>::TAG_OFFSET + 1;
    const TAG_TIME_OUT_ACK: Tag = MpiAnytimeSearch::<'a, T, N, M, E, B, F, V>::TAG_OFFSET + 2;
    const TAG_TERMINATION_DETECTION: Tag =
        MpiAnytimeSearch::<'a, T, N, M, E, B, F, V>::TAG_OFFSET + 3;
    const TAG_TERMINATE: Tag = MpiAnytimeSearch::<'a, T, N, M, E, B, F, V>::TAG_OFFSET + 4;

    /// Creates a new HDAPPS solver.
    pub fn new(
        input: SearchInput<'a, M, TransitionWithId<V>>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: MpiAnytimeSearchParameters<T>,
        progressive_parameters: ProgressiveSearchParameters,
        hash_function: F,
        communicator: &'a SimpleCommunicator,
    ) -> HdApps<'a, T, N, M, E, B, F, V> {
        let model = input.generator.model.clone();
        let mut search = MpiAnytimeSearch::new(
            input.generator,
            input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
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

        let open = BinaryHeap::new();
        let mut children = BinaryHeap::new();
        let suspend = BinaryHeap::new();

        search.generate_root_node(input.node, |node| {
            children.push((node, 0));
        });

        Self {
            model,
            search,
            communicator,
            node_communicator,
            progressive_parameters,
            width: progressive_parameters.init,
            open,
            children,
            suspend,
            current_depth: 0,
            is_time_out: false,
            n_remaining_time_out_ack: 0,
            is_checking_termination: false,
            is_terminated: false,
        }
    }

    fn receive_node(&mut self, source_rank: Rank) {
        self.search.increment_received();

        if let Some((node, depth)) = self
            .node_communicator
            .receive(source_rank, self.search.get_primal_bound())
        {
            let callback = |node| {
                if depth < self.current_depth {
                    self.suspend.push((node, depth));
                } else {
                    self.children.push((node, depth));
                }
            };

            let node = N::from(node);
            self.search.open_node(node, callback);
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
            || (!self.is_time_out
                && (!self.open.is_empty()
                    || !self.children.is_empty()
                    || !self.suspend.is_empty()));
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

    pub fn search(mut self) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        let mut goal_found = false;

        'outer: loop {
            self.process_message();

            if self.is_time_out || self.is_terminated {
                break 'outer;
            }

            if self.communicator.rank() == self.search.get_root_rank()
                && self.search.check_time_limit()
            {
                self.is_time_out = true;
                self.broadcast_time_out();
                break 'outer;
            }

            while let Some((node, depth)) = self.children.pop() {
                if node.is_closed() {
                    continue;
                }

                if let Some(bound) = node.bound(&self.model) {
                    if exceed_bound(&self.model, bound, self.search.get_primal_bound()) {
                        if N::ordered_by_bound() {
                            self.children.clear();
                        }

                        continue;
                    }
                }

                if self.open.len() < self.width {
                    self.open.push((node, depth));
                } else {
                    self.suspend.push((node, depth));
                }
            }

            if self.open.is_empty() && !self.suspend.is_empty() {
                if self.progressive_parameters.reset && goal_found {
                    self.width = self.progressive_parameters.init;
                } else {
                    self.width = self.progressive_parameters.increase_width(self.width);
                }

                let mut current_depth = usize::MAX;

                while self.open.len() < self.width {
                    if let Some((node, depth)) = self.suspend.pop() {
                        if node.is_closed() {
                            continue;
                        }
                        if let Some(bound) = node.bound(&self.model) {
                            if exceed_bound(&self.model, bound, self.search.get_primal_bound()) {
                                if N::ordered_by_bound() {
                                    self.suspend.clear();
                                }
                                continue;
                            }
                        }
                        current_depth = cmp::min(depth, current_depth);
                        self.open.push((node, depth));
                    } else {
                        break;
                    }
                }

                goal_found = false;
                self.current_depth = current_depth;
            } else if !self.open.is_empty() {
                self.current_depth += 1;
            }

            if self.communicator.rank() == self.search.get_root_rank()
                && self.open.is_empty()
                && !self.is_checking_termination
                && !self.search.cannot_terminate()
            {
                self.is_checking_termination = true;
                let destination_rank = (self.communicator.rank() + 1) % self.communicator.size();
                self.node_communicator
                    .initiate_termination(destination_rank);
            }

            while !self.open.is_empty() {
                self.process_message();

                if self.is_time_out || self.is_terminated {
                    break 'outer;
                }

                if self.communicator.rank() == self.search.get_root_rank()
                    && self.search.check_time_limit()
                {
                    self.is_time_out = true;
                    self.broadcast_time_out();
                    break 'outer;
                }

                let (node, depth) = self.open.pop().unwrap();

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
                    self.children.push((successor, depth + 1));
                };

                let send_callback = |destination_rank, successor| {
                    self.node_communicator
                        .send(destination_rank, &successor, depth + 1);
                };

                goal_found |= self.search.expand(node, local_callback, send_callback);
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

        let open_best = self.open.peek().and_then(|(n, _)| n.bound(&self.model));
        let children_best = self.children.peek().and_then(|(n, _)| n.bound(&self.model));
        let suspend_best = self.suspend.peek().and_then(|(n, _)| n.bound(&self.model));
        let dual_bound_array = [open_best, children_best, suspend_best];
        let dual_bound_iter = dual_bound_array.iter().filter_map(|x| *x);

        let dual_bound = if self.model.reduce_function == ReduceFunction::Max {
            dual_bound_iter.max()
        } else {
            dual_bound_iter.min()
        };

        let (mut solution, statistics) = self.search.finalize(dual_bound);

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

        solution.time = self.search.elapsed_time();

        (solution, statistics)
    }
}
