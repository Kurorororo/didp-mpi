use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure::{
    exceed_bound, Beam, HashableSignatureVariables,
};
use dypdl_heuristic_search::search_algorithm::util::{print_dual_bound, TimeKeeper};
use dypdl_heuristic_search::search_algorithm::{
    get_solution_cost_and_suffix, BeamSearchParameters, SearchInput, Solution, StateRegistry,
    SuccessorGenerator, TransitionWithId,
};
use dypdl_heuristic_search::{Parameters, ProgressiveSearchParameters};
use memoffset::offset_of;
use mpi::datatype::SystemDatatype;
use mpi::{
    datatype::{MutView, UserDatatype, View},
    topology::SimpleCommunicator,
    Rank, Tag,
};
use mpi::{traits::*, Address};
use std::collections::BinaryHeap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::mem;
use std::rc::Rc;
use std::str::FromStr;
use zerocopy::{AsBytes, FromBytes};

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::distributed_id_chain::{
    DistributedTransitionIdChain, GetDistributedTransitionIdChain, TransitionId,
};
use crate::is_float::IsFloat;
use crate::node_communicator::{self, TimeStampedNodeDepthCommunicator};
use crate::node_data_type::NodeDatatype;
use crate::partial_solution::{
    receive_partial_solution, receive_partial_solution_with_timestamp, send_partial_solution,
    send_partial_solution_with_time_stamp,
};
use crate::state_serializer::StateSerializer;
use crate::statistics::Statistics;
use crate::write_solution;

#[derive(Copy, Clone, Debug, Default)]
struct NTransitionIdsAndCost<T>(usize, T);

unsafe impl<T> Equivalence for NTransitionIdsAndCost<T>
where
    T: Equivalence<Out = SystemDatatype>,
{
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(NTransitionIdsAndCost<T>, 0) as Address,
                offset_of!(NTransitionIdsAndCost<T>, 1) as Address,
            ],
            &[usize::equivalent_datatype(), T::equivalent_datatype()],
        )
    }
}

pub struct MpiAnytimeSearch<'a, T, N, E, B, F, V = Transition>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T>,
    V: TransitionInterface + Clone + Default,
{
    communicator: &'a SimpleCommunicator,
    hash_function: F,
    generator: SuccessorGenerator<TransitionWithId<V>>,
    suffix: &'a [TransitionWithId<V>],
    transition_evaluator: E,
    base_cost_evaluator: B,
    primal_bound: Option<T>,
    quiet: bool,
    registry: StateRegistry<T, N>,
    id_to_chain_node: Vec<Rc<DistributedTransitionIdChain>>,
    time_keeper: TimeKeeper,
    solution: Solution<T, TransitionWithId<V>>,
    statistics: Statistics,
    reverse_transition_ids: Vec<TransitionId>,
    partial_solution_timestamp: usize,
}

impl<'a, T, N, E, B, F, V> MpiAnytimeSearch<'a, T, N, E, B, F, V>
where
    T: Numeric + IsFloat + Ord + Display + Equivalence<Out = SystemDatatype>,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    N: BfsNodeWithDistributedIdChain<T>,
    E: Fn(&N, &TransitionWithId<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: Fn(T, T) -> T,
    F: Fn(&HashableSignatureVariables) -> u64,
    V: TransitionInterface + Clone + Default,
    Transition: From<V> + From<TransitionWithId<V>>,
    TransitionWithId<V>: Clone,
{
    const TAG_PRIMAL_BOUND: Tag = 0;
    const TAG_PARTIAL_SOLUTION_REQUEST: Tag = 1;
    const TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA: Tag = 2;
    const TAG_PARTIAL_SOLUTION_TRANSITION_IDS: Tag = 3;
    const TAG_N_TRANSITION_IDS_AND_COST: Tag = 4;
    const TAG_REVERSE_TRANSITION_IDS: Tag = 5;
    pub const TAG_OFFSET: usize = 6;

    pub fn new(
        generator: SuccessorGenerator<TransitionWithId<V>>,
        suffix: &'a [TransitionWithId<V>],
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: Parameters<T>,
        hash_function: F,
        communicator: &'a SimpleCommunicator,
    ) -> MpiAnytimeSearch<'a, T, N, E, B, F, V> {
        let time_keeper = parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let quiet = parameters.quiet;

        let mut registry = StateRegistry::<_, _>::new(generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let solution = Solution::default();
        let statistics = Statistics::default();

        Self {
            communicator,
            hash_function,
            generator,
            suffix,
            transition_evaluator,
            base_cost_evaluator,
            primal_bound,
            quiet,
            registry,
            id_to_chain_node: Vec::default(),
            time_keeper,
            solution,
            statistics,
            reverse_transition_ids: Vec::default(),
            partial_solution_timestamp: 0,
        }
    }

    pub fn update_solution_transitions(&mut self, reverse_transition_ids: &[TransitionId]) {
        self.solution.transitions.clear();
        self.solution
            .transitions
            .extend(reverse_transition_ids.iter().rev().map(|id| {
                if id.1 {
                    self.generator.forced_transitions[id.0].as_ref().clone()
                } else {
                    self.generator.transitions[id.0].as_ref().clone()
                }
            }));

        if self.communicator.rank() == 0 {
            write_solution(&self.solution, "solution.yaml");
        }
    }

    pub fn send_solution(&self) {
        let destination = self.communicator.process_at_rank(0);
        let n = self.reverse_transition_ids.len();
        let cost = self.primal_bound.unwrap();
        let message = NTransitionIdsAndCost(n, cost);
        destination.buffered_send_with_tag(&message, Self::TAG_N_TRANSITION_IDS_AND_COST);
        destination.buffered_send_with_tag(
            &self.reverse_transition_ids,
            Self::TAG_REVERSE_TRANSITION_IDS,
        );
    }

    pub fn receive_solution(&mut self, source_rank: Rank) {
        let source = self.communicator.process_at_rank(source_rank);
        let mut message = NTransitionIdsAndCost::<T>::default();
        source.receive_into_with_tag(&mut message, Self::TAG_N_TRANSITION_IDS_AND_COST);
        let n = message.0;
        let cost = message.1;
        let mut tmp_transition_ids = vec![TransitionId::default(); n];
        source.receive_into_with_tag(&mut tmp_transition_ids, Self::TAG_REVERSE_TRANSITION_IDS);

        if !exceed_bound(&self.generator.model, cost, self.primal_bound) {
            self.update_solution_transitions(&tmp_transition_ids);
        }
    }

    pub fn broadcast_primal_bound(&mut self) {
        if let Some(primal_bound) = self.primal_bound {
            for destination_rank in 0..self.communicator.size() {
                if destination_rank != self.communicator.rank() {
                    let destination = self.communicator.process_at_rank(destination_rank);
                    destination.buffered_send_with_tag(&primal_bound, Self::TAG_PRIMAL_BOUND);
                }
            }
        }
    }

    pub fn receive_primal_bound(&mut self, source_rank: Rank) {
        let source = self.communicator.process_at_rank(source_rank);
        let mut primal_bound = T::default();
        source.receive_into_with_tag(&mut primal_bound, Self::TAG_PRIMAL_BOUND);

        if !exceed_bound(&self.generator.model, primal_bound, self.primal_bound) {
            self.primal_bound = Some(primal_bound);
        }
    }

    pub fn check_solution(&mut self, node: &N) {
        let model = &self.generator.model;

        if let Some((cost, suffix)) =
            get_solution_cost_and_suffix(model, node, self.suffix, &self.base_cost_evaluator)
        {
            if !exceed_bound(model, cost, self.primal_bound) {
                self.primal_bound = Some(cost);

                self.broadcast_primal_bound();

                self.reverse_transition_ids.clear();
                self.reverse_transition_ids
                    .extend(suffix.iter().rev().map(|t| TransitionId(t.id, t.forced)));
                let chain = node.get_distributed_transition_id_chain();
                let (additional_ids, parent) =
                    chain.get_transition_ids_in_this_rank(&self.id_to_chain_node);
                self.reverse_transition_ids.extend(additional_ids);

                if let Some((parent_rank, parent_id)) = parent {
                    self.partial_solution_timestamp += 1;
                    let buffer = [parent_id, self.partial_solution_timestamp];
                    self.communicator
                        .process_at_rank(parent_rank)
                        .buffered_send_with_tag(&buffer, Self::TAG_PARTIAL_SOLUTION_REQUEST);
                } else {
                    let reverse_transition_ids = self.reverse_transition_ids.clone();
                    self.update_solution_transitions(&reverse_transition_ids);
                    self.send_solution();
                }
            }
        }
    }

    pub fn receive_partial_solution_request(&mut self, source_rank: Rank) {
        let mut buffer = [0usize; 2];
        self.communicator
            .process_at_rank(source_rank)
            .receive_into_with_tag(&mut buffer, Self::TAG_PARTIAL_SOLUTION_REQUEST);
        let chain_id = buffer[0];
        let timestamp = buffer[1];

        let chain = &self.id_to_chain_node[chain_id];
        let (transition_ids, parent) =
            chain.get_transition_ids_in_this_rank(&self.id_to_chain_node);
        send_partial_solution_with_time_stamp(
            self.communicator,
            &transition_ids,
            parent,
            timestamp,
            source_rank,
            Self::TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA,
            Self::TAG_PARTIAL_SOLUTION_TRANSITION_IDS,
        )
    }

    pub fn receive_partial_solution_response(&mut self, source_rank: Rank) {
        let (parent, up_to_date) = receive_partial_solution_with_timestamp(
            self.communicator,
            &mut self.reverse_transition_ids,
            source_rank,
            self.partial_solution_timestamp,
            Self::TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA,
            Self::TAG_PARTIAL_SOLUTION_TRANSITION_IDS,
        );

        if up_to_date {
            if let Some((parent_rank, parent_id)) = parent {
                self.partial_solution_timestamp += 1;
                let buffer = [parent_id, self.partial_solution_timestamp];
                self.communicator
                    .process_at_rank(parent_rank)
                    .buffered_send_with_tag(&buffer, Self::TAG_PARTIAL_SOLUTION_REQUEST);
            } else {
                let reverse_transition_ids = self.reverse_transition_ids.clone();
                self.update_solution_transitions(&reverse_transition_ids);
                self.send_solution();
            }
        }
    }
}
