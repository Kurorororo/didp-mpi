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
use crate::partial_solution::{receive_partial_solution, send_partial_solution};
use crate::state_serializer::StateSerializer;
use crate::statistics::Statistics;
use crate::write_solution;

pub struct HdAcps<'a, T, N, M, E, B, F, V = Transition>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    V: TransitionInterface + Clone + Default,
{
    communicator: &'a SimpleCommunicator,
    node_communicator: TimeStampedNodeDepthCommunicator<'a, SimpleCommunicator, M, T>,
    hash_function: F,
    generator: SuccessorGenerator<TransitionWithId<V>>,
    suffix: &'a [TransitionWithId<V>],
    transition_evaluator: E,
    base_cost_evaluator: B,
    progressive_parameters: ProgressiveSearchParameters,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    width: usize,
    open: Vec<BinaryHeap<Rc<N>>>,
    registry: StateRegistry<T, N>,
    id_to_chain_node: Vec<Rc<DistributedTransitionIdChain>>,
    layer_index: usize,
    node_index: usize,
    no_node: bool,
    goal_found: bool,
    dual_bound_candidate: Option<T>,
    time_keeper: TimeKeeper,
    solution: Solution<T, TransitionWithId<V>>,
    statistics: Statistics,
    reverse_transition_ids: Vec<TransitionId>,
    partial_solution_timestamp: usize,
    _phantom: PhantomData<M>,
}

impl<'a, T, N, M, E, B, F, V> HdAcps<'a, T, N, M, E, B, F, V>
where
    T: Numeric + IsFloat + Ord + Display,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    M: Clone + NodeDatatype<T>,
    E: Fn(&N, &TransitionWithId<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: Fn(T, T) -> T,
    F: Fn(&HashableSignatureVariables) -> u64,
    V: TransitionInterface + Clone + Default,
    Transition: From<V> + From<TransitionWithId<V>>,
    TransitionWithId<V>: Clone,
{
    const TAG_NODE: Tag = 0;
    const TAG_PRIMAL_BOUND: Tag = 1;
    const TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA: Tag = 2;
    const TAG_PARTIAL_SOLUTION_TRANSITION_IDS: Tag = 3;
    const TAG_TERMINATION_DETECTION: Tag = 4;
    const TAG_N_TRANSITION_IDS: Tag = 5;
    const TAG_REVERSE_TRANSITION_IDS: Tag = 6;
    const TAG_TERMINATE: Tag = 7;

    /// Creates a new HDACPS solver.
    pub fn new(
        input: SearchInput<'a, M, TransitionWithId<V>>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: Parameters<T>,
        progressive_parameters: ProgressiveSearchParameters,
        hash_function: F,
        communicator: &'a SimpleCommunicator,
    ) -> HdAcps<'a, T, N, M, E, B, F, V> {
        let mut time_keeper = parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let quiet = parameters.quiet;
        let n_ranks = communicator.size();

        let mut open = vec![BinaryHeap::new()];
        let mut registry = StateRegistry::<_, _>::new(input.generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();
        let mut statistics = Statistics::default();

        if let Some(node) = input.node {
            let hash_value = hash_function(node.get_signature());
            let assigned_rank = (hash_value % n_ranks as u64) as Rank;

            if assigned_rank == communicator.rank() {
                let node = N::from(node);
                let (node, _) = registry.insert(node).unwrap();
                solution.generated += 1;
                statistics.generated += 1;
                solution.best_bound = node.bound(&input.generator.model);
                open[0].push(node);

                if !quiet {
                    solution.time = time_keeper.elapsed_time();
                    print_dual_bound(&solution);
                }
            }
        } else {
            solution.is_infeasible = true;
        }

        let node_communicator = TimeStampedNodeDepthCommunicator::new(
            communicator,
            Self::TAG_NODE,
            input.generator.model.clone(),
        );
        time_keeper.stop();

        Self {
            communicator,
            node_communicator,
            hash_function,
            generator: input.generator,
            suffix: input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
            progressive_parameters,
            primal_bound,
            get_all_solutions: parameters.get_all_solutions,
            quiet,
            width: progressive_parameters.init,
            open,
            registry,
            id_to_chain_node: Vec::default(),
            layer_index: 0,
            node_index: 0,
            no_node: true,
            goal_found: false,
            dual_bound_candidate: None,
            time_keeper,
            solution,
            statistics,
            reverse_transition_ids: Vec::default(),
            partial_solution_timestamp: 0,
            _phantom: PhantomData,
        }
    }

    pub fn update_solution_transitions(&mut self, reverse_transition_ids: &[TransitionId]) {
        self.solution.transitions = reverse_transition_ids
            .iter()
            .rev()
            .map(|id| {
                if id.1 {
                    self.generator.forced_transitions[id.0].as_ref().clone()
                } else {
                    self.generator.transitions[id.0].as_ref().clone()
                }
            })
            .collect::<Vec<_>>();

        if self.communicator.rank() == 0 {
            write_solution(&self.solution, "solution.yaml");
        }
    }

    pub fn send_solution(&self) {
        let destination = self.communicator.process_at_rank(0);
        let n = self.reverse_transition_ids.len();
        destination.buffered_send_with_tag(&n, Self::TAG_N_TRANSITION_IDS);
        destination.buffered_send_with_tag(
            &self.reverse_transition_ids,
            Self::TAG_REVERSE_TRANSITION_IDS,
        );
    }

    pub fn receive_solution(&mut self, source_rank: Rank) {
        let source = self.communicator.process_at_rank(source_rank);
        let mut n = 0;
        source.receive_into_with_tag(&mut n, Self::TAG_N_TRANSITION_IDS);
        let mut tmp_transition_ids = vec![TransitionId::default(); n];
        source.receive_into_with_tag(&mut tmp_transition_ids, Self::TAG_REVERSE_TRANSITION_IDS);
        self.update_solution_transitions(&tmp_transition_ids);
    }

    pub fn notify_primal_bound_and_request_partial_solution(&mut self, node: &N) {
        let chain = node.get_distributed_transition_id_chain();
        let (transition_ids, parent) =
            chain.get_transition_ids_in_this_rank(&self.id_to_chain_node);
        self.reverse_transition_ids = transition_ids;

        if let Some((parent_rank, parent_id)) = parent {
        } else {
        }
    }

    pub fn search(&mut self) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        (self.solution.clone(), Vec::default())
    }
}
