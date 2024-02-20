use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure::{
    exceed_bound, HashableSignatureVariables,
};
use dypdl_heuristic_search::search_algorithm::{
    Solution, StateRegistry, SuccessorGenerator, TransitionWithId,
};
use dypdl_heuristic_search::Parameters;
use memoffset::offset_of;
use mpi::datatype::SystemDatatype;
use mpi::{datatype::UserDatatype, topology::SimpleCommunicator, Rank, Tag};
use mpi::{traits::*, Address};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;
use std::str::FromStr;

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::distributed_id_chain::DistributedTransitionIdChain;
use crate::is_float::IsFloat;
use crate::node_data_type::NodeDatatype;
use crate::partial_solution::{
    receive_partial_solution_with_timestamp, send_partial_solution_with_time_stamp,
    PartialSolutionTags,
};
use crate::statistics::Statistics;
use crate::write_solution;

#[derive(Copy, Clone, Debug, Default)]
struct NTransitionIdsAndCostForSend<T>(usize, T);

unsafe impl<T> Equivalence for NTransitionIdsAndCostForSend<T>
where
    T: Equivalence<Out = SystemDatatype>,
{
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(NTransitionIdsAndCostForSend<T>, 0) as Address,
                offset_of!(NTransitionIdsAndCostForSend<T>, 1) as Address,
            ],
            &[usize::equivalent_datatype(), T::equivalent_datatype()],
        )
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct NTransitionIdsAndCost<T> {
    pub n_transitions: usize,
    pub cost: T,
}

impl<T: IsFloat> NTransitionIdsAndCost<T> {
    pub fn send<D>(&self, destination: &D, tag: Tag)
    where
        D: Destination,
    {
        if T::is_float() {
            let message = NTransitionIdsAndCostForSend::<Continuous>(
                self.n_transitions,
                self.cost.to_continuous(),
            );
            destination.buffered_send_with_tag(&message, tag);
        } else {
            let message =
                NTransitionIdsAndCostForSend::<Integer>(self.n_transitions, self.cost.to_integer());
            destination.buffered_send_with_tag(&message, tag);
        }
    }

    pub fn receive<S>(source: &S, tag: Tag) -> Self
    where
        S: Source,
    {
        if T::is_float() {
            let (message, _) =
                source.receive_with_tag::<NTransitionIdsAndCostForSend<Continuous>>(tag);
            Self {
                n_transitions: message.0,
                cost: T::from(message.1),
            }
        } else {
            let (message, _) =
                source.receive_with_tag::<NTransitionIdsAndCostForSend<Integer>>(tag);
            Self {
                n_transitions: message.0,
                cost: T::from(message.1),
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct FinalSolutionInformationForSend<T>([bool; 2], [T; 2], usize);

unsafe impl<T> Equivalence for FinalSolutionInformationForSend<T>
where
    T: Equivalence<Out = SystemDatatype>,
{
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[2, 2, 1],
            &[
                offset_of!(FinalSolutionInformationForSend<T>, 0) as Address,
                offset_of!(FinalSolutionInformationForSend<T>, 1) as Address,
                offset_of!(FinalSolutionInformationForSend<T>, 2) as Address,
            ],
            &[
                bool::equivalent_datatype(),
                T::equivalent_datatype(),
                usize::equivalent_datatype(),
            ],
        )
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct FinalSolutionInformation<T> {
    pub cost: Option<T>,
    pub bound: Option<T>,
    pub n_transitions: Option<usize>,
}

impl<T: IsFloat> FinalSolutionInformation<T> {
    pub fn send<D: Destination>(&self, destination: &D, tag: Tag) {
        let has_cost = self.cost.is_some();
        let has_bound = self.bound.is_some();
        let n_transitions = self.n_transitions.unwrap_or(0);

        if T::is_float() {
            let cost = self.cost.map_or(0.0, |cost| cost.to_continuous());
            let bound = self.bound.map_or(0.0, |bound| bound.to_continuous());

            let message = FinalSolutionInformationForSend::<Continuous>(
                [has_cost, has_bound],
                [cost, bound],
                n_transitions,
            );
            destination.buffered_send_with_tag(&message, tag);
        } else {
            let cost = self.cost.map_or(0, |cost| cost.to_integer());
            let bound = self.bound.map_or(0, |bound| bound.to_integer());

            let message = FinalSolutionInformationForSend::<Integer>(
                [has_cost, has_bound],
                [cost, bound],
                n_transitions,
            );
            destination.buffered_send_with_tag(&message, tag);
        }
    }

    pub fn receive<S: Source>(source: &S, tag: Tag) -> Self {
        if T::is_float() {
            let (message, _) =
                source.receive_with_tag::<FinalSolutionInformationForSend<Continuous>>(tag);
            Self {
                cost: if message.0[0] {
                    Some(T::from(message.1[0]))
                } else {
                    None
                },
                bound: if message.0[1] {
                    Some(T::from(message.1[1]))
                } else {
                    None
                },
                n_transitions: if message.0[0] { Some(message.2) } else { None },
            }
        } else {
            let (message, _) =
                source.receive_with_tag::<FinalSolutionInformationForSend<Integer>>(tag);
            Self {
                cost: if message.0[0] {
                    Some(T::from(message.1[0]))
                } else {
                    None
                },
                bound: if message.0[1] {
                    Some(T::from(message.1[1]))
                } else {
                    None
                },
                n_transitions: if message.0[0] { Some(message.2) } else { None },
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MpiAnytimeSearchParameters<T>
where
    T: Numeric,
{
    pub controller_rank: Rank,
    pub solution_filename: Option<String>,
    pub parameters: Parameters<T>,
}

struct MpiSolutionManager<'a, T, B, V>
where
    T: Numeric,
    V: TransitionInterface + Clone + Default,
{
    model: Rc<Model>,
    forced_transitions: Vec<TransitionWithId<V>>,
    transitions: Vec<TransitionWithId<V>>,
    communicator: &'a SimpleCommunicator,
    root_rank: Rank,
    base_cost_evaluator: B,
    suffix: &'a [TransitionWithId<V>],
    primal_bound: Option<T>,
    solution: Solution<T, TransitionWithId<V>>,
    statistics: Statistics,
    local_solution_cost: Option<T>,
    reverse_transition_ids: Vec<usize>,
    reverse_transition_forced: Vec<bool>,
    is_retrieving_partial_solution: bool,
    partial_solution_timestamp: usize,
    n_partial_solution_remaining: usize,
    n_primal_bound_ack_remaining: usize,
    n_solution_ack_remaining: usize,
    solution_filename: Option<String>,
    quiet: bool,
}

impl<'a, T, B, V> MpiSolutionManager<'a, T, B, V>
where
    T: Numeric + IsFloat + Ord + Display,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V> + From<TransitionWithId<V>>,
    TransitionWithId<V>: Clone,
{
    const TAG_PRIMAL_BOUND: Tag = 0;
    const TAG_PRIMAL_BOUND_ACK: Tag = 1;
    const TAG_PARTIAL_SOLUTION_REQUEST: Tag = 2;
    const TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA: Tag = 3;
    const TAG_PARTIAL_SOLUTION: PartialSolutionTags = PartialSolutionTags {
        fixed_length_data: 3,
        transition_ids: 4,
        transition_forced: 5,
    };
    const TAG_N_TRANSITION_IDS_AND_COST: Tag = 6;
    const TAG_REVERSE_TRANSITION_IDS: Tag = 7;
    const TAG_REVERSE_TRANSITION_FORCED: Tag = 8;
    const TAG_SOLUTION_ACK: Tag = 9;
    const TAG_FINAL_SOLUTION_INFORMATION: Tag = 10;
    const TAG_FINAL_TRANSITION_IDS_REQUEST: Tag = 11;
    const TAG_FINAL_TRANSITION_IDS: Tag = 12;
    const TAG_FINAL_TRANSITION_FORCED: Tag = 13;
    pub const TAG_OFFSET: Tag = 14;

    pub fn new(
        generator: &SuccessorGenerator<TransitionWithId<V>>,
        suffix: &'a [TransitionWithId<V>],
        base_cost_evaluator: B,
        parameters: MpiAnytimeSearchParameters<T>,
        communicator: &'a SimpleCommunicator,
    ) -> Self {
        let model = generator.model.clone();
        let forced_transitions = generator
            .forced_transitions
            .iter()
            .map(|t| t.as_ref().clone())
            .collect();
        let transitions = generator
            .transitions
            .iter()
            .map(|t| t.as_ref().clone())
            .collect();
        let root_rank = parameters.controller_rank;
        let solution_filename = parameters.solution_filename;
        let primal_bound = parameters.parameters.primal_bound;
        let quiet = parameters.parameters.quiet;

        Self {
            model,
            forced_transitions,
            transitions,
            communicator,
            root_rank,
            base_cost_evaluator,
            suffix,
            primal_bound,
            solution: Solution::default(),
            statistics: Statistics::default(),
            local_solution_cost: None,
            reverse_transition_ids: Vec::default(),
            reverse_transition_forced: Vec::default(),
            is_retrieving_partial_solution: false,
            partial_solution_timestamp: 0,
            n_partial_solution_remaining: 0,
            n_primal_bound_ack_remaining: 0,
            n_solution_ack_remaining: 0,
            solution_filename,
            quiet,
        }
    }

    pub fn get_root_rank(&self) -> Rank {
        self.root_rank
    }

    pub fn get_primal_bound(&self) -> Option<T> {
        self.primal_bound
    }

    pub fn cannot_terminate(&self) -> bool {
        self.is_retrieving_partial_solution
            || self.n_partial_solution_remaining > 0
            || self.n_primal_bound_ack_remaining > 0
            || self.n_solution_ack_remaining > 0
    }

    pub fn is_quiet(&self) -> bool {
        self.quiet
    }

    pub fn increment_expanded(&mut self) {
        self.solution.expanded += 1;
        self.statistics.expanded += 1;
    }

    pub fn increment_generated(&mut self) {
        self.solution.generated += 1;
        self.statistics.generated += 1;
    }

    pub fn increment_sent(&mut self) {
        self.statistics.sent += 1;
    }

    pub fn increment_received(&mut self) {
        self.statistics.received += 1;
    }

    pub fn increment_kept(&mut self) {
        self.statistics.kept += 1;
    }

    pub fn update_dual_bound(&mut self, dual_bound: T) {
        self.solution.best_bound = Some(dual_bound);

        if !self.quiet {
            println!("New dual bound: {}", dual_bound,);
        }
    }

    fn update_solution_transitions(
        &mut self,
        reverse_transition_ids: &[usize],
        reverse_transition_forced: &[bool],
    ) {
        self.solution.transitions.clear();
        self.solution.transitions.extend(
            reverse_transition_ids
                .iter()
                .zip(reverse_transition_forced.iter())
                .rev()
                .map(|(id, forced)| {
                    if *forced {
                        self.forced_transitions[*id].clone()
                    } else {
                        self.transitions[*id].clone()
                    }
                }),
        );

        if let Some(filename) = self.solution_filename.as_ref() {
            write_solution(&self.solution, filename);
        }
    }

    fn send_solution(&mut self) {
        let n = self.reverse_transition_ids.len();
        let cost = self.primal_bound.unwrap();
        let message = NTransitionIdsAndCost {
            n_transitions: n,
            cost,
        };
        let destination = self.communicator.process_at_rank(self.root_rank);
        message.send(&destination, Self::TAG_N_TRANSITION_IDS_AND_COST);
        destination.buffered_send_with_tag(
            &self.reverse_transition_ids,
            Self::TAG_REVERSE_TRANSITION_IDS,
        );
        destination.buffered_send_with_tag(
            &self.reverse_transition_forced,
            Self::TAG_REVERSE_TRANSITION_FORCED,
        );
        self.n_solution_ack_remaining += 1;
    }

    fn receive_solution(&mut self, source_rank: Rank) {
        let source = self.communicator.process_at_rank(source_rank);
        let message =
            NTransitionIdsAndCost::<T>::receive(&source, Self::TAG_N_TRANSITION_IDS_AND_COST);
        let n = message.n_transitions;
        let cost = message.cost;
        let mut tmp_transition_ids = vec![0; n];
        let mut tmp_transition_forced = vec![false; n];
        source.receive_into_with_tag(&mut tmp_transition_ids, Self::TAG_REVERSE_TRANSITION_IDS);
        source.receive_into_with_tag(
            &mut tmp_transition_forced,
            Self::TAG_REVERSE_TRANSITION_FORCED,
        );

        if !exceed_bound(&self.model, cost, self.primal_bound) {
            self.solution.cost = Some(cost);
            self.update_solution_transitions(&tmp_transition_ids, &tmp_transition_forced);
        }

        let buffer: [u8; 0] = [];
        source.buffered_send_with_tag(&buffer, Self::TAG_SOLUTION_ACK);
    }

    fn receive_solution_ack(&mut self, source_rank: Rank) {
        let source = self.communicator.process_at_rank(source_rank);
        let mut buffer: [u8; 0] = [];
        source.receive_into_with_tag(&mut buffer, Self::TAG_SOLUTION_ACK);
        self.n_solution_ack_remaining -= 1;
    }

    fn broadcast_primal_bound(&mut self) {
        if let Some(primal_bound) = self.primal_bound {
            if T::is_float() {
                let primal_bound = primal_bound.to_continuous();

                for destination_rank in 0..self.communicator.size() {
                    if destination_rank != self.communicator.rank() {
                        let destination = self.communicator.process_at_rank(destination_rank);
                        destination.buffered_send_with_tag(&primal_bound, Self::TAG_PRIMAL_BOUND);
                        self.n_primal_bound_ack_remaining += 1;
                    }
                }
            } else {
                let primal_bound = primal_bound.to_integer();

                for destination_rank in 0..self.communicator.size() {
                    if destination_rank != self.communicator.rank() {
                        let destination = self.communicator.process_at_rank(destination_rank);
                        destination.buffered_send_with_tag(&primal_bound, Self::TAG_PRIMAL_BOUND);
                        self.n_primal_bound_ack_remaining += 1;
                    }
                }
            }

            if !self.quiet {
                println!("New primal bound: {}", primal_bound,);
            }
        }
    }

    fn receive_primal_bound(&mut self, source_rank: Rank) {
        let source = self.communicator.process_at_rank(source_rank);

        let primal_bound = if T::is_float() {
            let (primal_bound, _) = source.receive_with_tag::<Continuous>(Self::TAG_PRIMAL_BOUND);
            T::from(primal_bound)
        } else {
            let (primal_bound, _) = source.receive_with_tag::<Integer>(Self::TAG_PRIMAL_BOUND);
            T::from(primal_bound)
        };

        if !exceed_bound(&self.model, primal_bound, self.primal_bound) {
            self.primal_bound = Some(primal_bound);

            if !self.quiet {
                println!("New primal bound: {}", primal_bound);
            }
        }

        let buffer: [u8; 0] = [];
        source.buffered_send_with_tag(&buffer, Self::TAG_PRIMAL_BOUND_ACK);
    }

    fn receive_primal_bound_ack(&mut self, source_rank: Rank) {
        let source = self.communicator.process_at_rank(source_rank);
        let mut buffer: [u8; 0] = [];
        source.receive_into_with_tag(&mut buffer, Self::TAG_PRIMAL_BOUND_ACK);
        self.n_primal_bound_ack_remaining -= 1;
    }

    fn check_solution<M>(
        &mut self,
        node: &M,
        id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
    ) -> (bool, bool)
    where
        M: NodeDatatype<T>,
    {
        if let Some((cost, suffix)) = node.get_solution_cost_and_suffix(
            &self.model,
            self.suffix,
            &mut self.base_cost_evaluator,
        ) {
            if !exceed_bound(&self.model, cost, self.primal_bound) {
                self.primal_bound = Some(cost);
                self.local_solution_cost = Some(cost);

                self.broadcast_primal_bound();

                self.reverse_transition_ids.clear();
                self.reverse_transition_ids
                    .extend(suffix.iter().rev().map(|t| t.id));
                self.reverse_transition_forced
                    .extend(suffix.iter().rev().map(|t| t.forced));
                let chain = node.get_distributed_transition_id_chain();
                let (additional_ids, additional_forced, parent) =
                    chain.get_transition_ids_in_this_rank(id_to_chain_node);
                self.reverse_transition_ids.extend(additional_ids);
                self.reverse_transition_forced.extend(additional_forced);

                if let Some((parent_rank, parent_id)) = parent {
                    self.is_retrieving_partial_solution = true;
                    self.partial_solution_timestamp += 1;
                    let buffer = [parent_id, self.partial_solution_timestamp];
                    let destination_process = self.communicator.process_at_rank(parent_rank);
                    destination_process
                        .buffered_send_with_tag(&buffer, Self::TAG_PARTIAL_SOLUTION_REQUEST);
                    self.n_partial_solution_remaining += 1;
                } else {
                    let reverse_transition_ids = self.reverse_transition_ids.clone();
                    let reverse_transition_forced = self.reverse_transition_forced.clone();
                    self.solution.cost = self.local_solution_cost;
                    self.update_solution_transitions(
                        &reverse_transition_ids,
                        &reverse_transition_forced,
                    );

                    if self.communicator.rank() != self.root_rank {
                        self.send_solution();
                    }
                }

                (true, true)
            } else {
                (true, false)
            }
        } else {
            (false, false)
        }
    }

    fn receive_partial_solution_request(
        &mut self,
        source_rank: Rank,
        id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
    ) {
        let mut buffer = [0usize; 2];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, Self::TAG_PARTIAL_SOLUTION_REQUEST);
        let chain_id = buffer[0];
        let timestamp = buffer[1];

        let chain = &id_to_chain_node[chain_id];
        let (transition_ids, transition_forced, parent) =
            chain.get_transition_ids_in_this_rank(id_to_chain_node);
        send_partial_solution_with_time_stamp(
            &source_process,
            &transition_ids,
            &transition_forced,
            parent,
            timestamp,
            &Self::TAG_PARTIAL_SOLUTION,
        )
    }

    fn receive_partial_solution_response(&mut self, source_rank: Rank) {
        let source_process = self.communicator.process_at_rank(source_rank);
        let (parent, up_to_date) = receive_partial_solution_with_timestamp(
            &source_process,
            &mut self.reverse_transition_ids,
            &mut self.reverse_transition_forced,
            self.partial_solution_timestamp,
            &Self::TAG_PARTIAL_SOLUTION,
        );
        self.n_partial_solution_remaining -= 1;

        if up_to_date {
            if let Some((parent_rank, parent_id)) = parent {
                let buffer = [parent_id, self.partial_solution_timestamp];
                let destination_process = self.communicator.process_at_rank(parent_rank);
                destination_process
                    .buffered_send_with_tag(&buffer, Self::TAG_PARTIAL_SOLUTION_REQUEST);
                self.n_partial_solution_remaining += 1;
            } else {
                self.is_retrieving_partial_solution = false;
                let reverse_transition_ids = self.reverse_transition_ids.clone();
                let reverse_transition_forced = self.reverse_transition_forced.clone();
                self.solution.cost = self.local_solution_cost;
                self.update_solution_transitions(
                    &reverse_transition_ids,
                    &reverse_transition_forced,
                );

                if self.communicator.rank() != self.root_rank {
                    self.send_solution();
                }
            }
        }
    }

    pub fn receive_message(
        &mut self,
        source_rank: Rank,
        tag: Tag,
        id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
    ) {
        match tag {
            Self::TAG_PRIMAL_BOUND => self.receive_primal_bound(source_rank),
            Self::TAG_PRIMAL_BOUND_ACK => self.receive_primal_bound_ack(source_rank),
            Self::TAG_PARTIAL_SOLUTION_REQUEST => {
                self.receive_partial_solution_request(source_rank, id_to_chain_node)
            }
            Self::TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA => {
                self.receive_partial_solution_response(source_rank)
            }
            Self::TAG_N_TRANSITION_IDS_AND_COST => self.receive_solution(source_rank),
            Self::TAG_SOLUTION_ACK => self.receive_solution_ack(source_rank),
            _ => {}
        }
    }

    pub fn finalize(
        &mut self,
        local_dual_bound: Option<T>,
    ) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        if self.communicator.rank() == self.root_rank {
            let model = &self.model;
            let n_ranks = self.communicator.size();
            let this_rank = self.communicator.rank();
            let mut best_rank = this_rank;
            let mut n_best_transitions = 0;
            let mut global_dual_bound = local_dual_bound;

            for source_rank in 0..n_ranks {
                if source_rank == this_rank {
                    continue;
                }

                let source = self.communicator.process_at_rank(source_rank);
                let information = FinalSolutionInformation::<T>::receive(
                    &source,
                    Self::TAG_FINAL_SOLUTION_INFORMATION,
                );

                if let Some(cost) = information.cost {
                    if !exceed_bound(model, cost, self.solution.cost) {
                        self.primal_bound = Some(cost);
                        self.solution.cost = Some(cost);
                        best_rank = source_rank;
                        n_best_transitions = information.n_transitions.unwrap();
                    }
                }

                if let Some(dual_bound) = information.bound {
                    if !exceed_bound(model, dual_bound, global_dual_bound) {
                        global_dual_bound = Some(dual_bound);
                    }
                }
            }

            self.solution.best_bound = global_dual_bound;

            for destination_rank in 0..n_ranks {
                if destination_rank == this_rank {
                    continue;
                }

                let request = destination_rank == best_rank;
                let destination_process = self.communicator.process_at_rank(destination_rank);
                destination_process.send_with_tag(&request, Self::TAG_FINAL_TRANSITION_IDS_REQUEST);
            }

            if best_rank != this_rank {
                let mut transition_ids = vec![0; n_best_transitions];
                let mut transition_forced = vec![false; n_best_transitions];
                let source_process = self.communicator.process_at_rank(best_rank);
                source_process
                    .receive_into_with_tag(&mut transition_ids, Self::TAG_FINAL_TRANSITION_IDS);
                source_process.receive_into_with_tag(
                    &mut transition_forced,
                    Self::TAG_FINAL_TRANSITION_FORCED,
                );
                self.solution.transitions.clear();
                self.solution.transitions.extend(
                    transition_ids
                        .into_iter()
                        .zip(transition_forced)
                        .map(|(id, forced)| {
                            if forced {
                                self.forced_transitions[id].clone()
                            } else {
                                self.transitions[id].clone()
                            }
                        }),
                );
            }

            let statistics = self
                .statistics
                .gather(self.communicator, self.root_rank, true);
            self.solution.expanded = 0;
            self.solution.generated = 0;

            for s in &statistics {
                self.solution.expanded += s.expanded;
                self.solution.generated += s.generated;
            }

            if self.solution.cost.is_some() {
                if let Some(filename) = self.solution_filename.as_ref() {
                    write_solution(&self.solution, filename);
                }
            }

            (self.solution.clone(), statistics)
        } else {
            let root = self.communicator.process_at_rank(self.root_rank);

            let information = FinalSolutionInformation {
                cost: self.solution.cost,
                bound: local_dual_bound,
                n_transitions: if self.solution.cost.is_some() {
                    Some(self.solution.transitions.len())
                } else {
                    None
                },
            };
            information.send(&root, Self::TAG_FINAL_SOLUTION_INFORMATION);

            let (request, _) =
                root.receive_with_tag::<bool>(Self::TAG_FINAL_TRANSITION_IDS_REQUEST);

            if request {
                let transition_ids = self
                    .solution
                    .transitions
                    .iter()
                    .map(|t| t.id)
                    .collect::<Vec<_>>();
                let transition_forced = self
                    .solution
                    .transitions
                    .iter()
                    .map(|t| t.forced)
                    .collect::<Vec<_>>();
                root.send_with_tag(&transition_ids[..], Self::TAG_FINAL_TRANSITION_IDS);
                root.send_with_tag(&transition_forced[..], Self::TAG_FINAL_TRANSITION_FORCED);
            }

            let statistics = self
                .statistics
                .gather(self.communicator, self.root_rank, false);

            (self.solution.clone(), statistics)
        }
    }
}

pub struct MpiAnytimeSearch<'a, T, N, M, E, B, F, V = Transition>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T>,
    V: TransitionInterface + Clone + Default,
{
    communicator: &'a SimpleCommunicator,
    hash_function: F,
    generator: SuccessorGenerator<TransitionWithId<V>>,
    transition_evaluator: E,
    registry: StateRegistry<T, N>,
    id_to_chain_node: Vec<Rc<DistributedTransitionIdChain>>,
    solution_manager: MpiSolutionManager<'a, T, B, V>,
    _phantom: PhantomData<M>,
}

impl<'a, T, N, M, E, B, F, V> MpiAnytimeSearch<'a, T, N, M, E, B, F, V>
where
    T: Numeric + IsFloat + Ord + Display,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    M: NodeDatatype<T>,
    E: FnMut(&N, &TransitionWithId<V>, Option<T>) -> Option<M>,
    B: FnMut(T, T) -> T,
    F: FnMut(&HashableSignatureVariables) -> u64,
    V: TransitionInterface + Clone + Default + 'static,
    Transition: From<V> + From<TransitionWithId<V>>,
    TransitionWithId<V>: Clone,
{
    pub const TAG_OFFSET: Tag = MpiSolutionManager::<'a, T, B, V>::TAG_OFFSET;

    pub fn new(
        generator: SuccessorGenerator<TransitionWithId<V>>,
        suffix: &'a [TransitionWithId<V>],
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: MpiAnytimeSearchParameters<T>,
        hash_function: F,
        communicator: &'a SimpleCommunicator,
    ) -> Self {
        let mut registry = StateRegistry::<_, _>::new(generator.model.clone());

        if let Some(capacity) = parameters.parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let solution_manager = MpiSolutionManager::new(
            &generator,
            suffix,
            base_cost_evaluator,
            parameters,
            communicator,
        );

        Self {
            communicator,
            hash_function,
            generator,
            transition_evaluator,
            registry,
            id_to_chain_node: Vec::default(),
            solution_manager,
            _phantom: PhantomData,
        }
    }

    pub fn get_root_rank(&self) -> Rank {
        self.solution_manager.get_root_rank()
    }

    pub fn is_quiet(&self) -> bool {
        self.solution_manager.is_quiet()
    }

    pub fn get_primal_bound(&self) -> Option<T> {
        self.solution_manager.get_primal_bound()
    }

    pub fn cannot_terminate(&self) -> bool {
        self.solution_manager.cannot_terminate()
    }

    fn open_node_inner<C>(
        node: N,
        mut callback: C,
        registry: &mut StateRegistry<T, N>,
        solution_manager: &mut MpiSolutionManager<'a, T, B, V>,
    ) -> bool
    where
        C: FnMut(Rc<N>),
    {
        if let Some((node, dominated)) = registry.insert(node) {
            if let Some(dominated) = dominated {
                if !dominated.is_closed() {
                    dominated.close();
                }
            } else {
                solution_manager.increment_generated();
            };

            callback(node);

            true
        } else {
            false
        }
    }

    pub fn open_node<C>(&mut self, node: N, callback: C) -> bool
    where
        C: FnMut(Rc<N>),
    {
        Self::open_node_inner(
            node,
            callback,
            &mut self.registry,
            &mut self.solution_manager,
        )
    }

    pub fn generate_root_node<C>(&mut self, node: Option<M>, callback: C)
    where
        C: FnMut(Rc<N>),
    {
        if let Some(node) = node {
            let hash_value = (self.hash_function)(node.get_signature());
            let assigned_rank = (hash_value % self.communicator.size() as u64) as Rank;

            if assigned_rank == self.communicator.rank() {
                self.solution_manager.increment_generated();

                let (is_goal, _) = self
                    .solution_manager
                    .check_solution(&node, &self.id_to_chain_node);

                if is_goal {
                    self.solution_manager.solution.is_optimal = true;
                } else {
                    let node = N::from(node);

                    if let Some(bound) = node.bound(&self.generator.model) {
                        self.solution_manager.update_dual_bound(bound);
                    }

                    self.open_node(node, callback);
                }
            }
        } else {
            self.solution_manager.solution.is_infeasible = true;
        }
    }

    pub fn increment_received(&mut self) {
        self.solution_manager.increment_received()
    }

    pub fn receive_message(&mut self, source_rank: Rank, tag: Tag) {
        self.solution_manager
            .receive_message(source_rank, tag, &self.id_to_chain_node)
    }

    pub fn expand<L, S>(&mut self, node: Rc<N>, mut local_callback: L, mut send_callback: S) -> bool
    where
        L: FnMut(Rc<N>),
        S: FnMut(Rank, M),
    {
        let mut no_successor = true;
        node.get_distributed_transition_id_chain()
            .id
            .set(Some(self.id_to_chain_node.len()));
        self.id_to_chain_node
            .push(node.get_distributed_transition_id_chain().clone());

        self.solution_manager.increment_expanded();
        let n_ranks = self.communicator.size() as u64;
        let this_rank = self.communicator.rank();
        let mut better_goal_found = false;

        for transition in self.generator.applicable_transitions(node.state()) {
            if let Some(successor) = (self.transition_evaluator)(
                node.as_ref(),
                transition.as_ref(),
                self.solution_manager.get_primal_bound(),
            ) {
                let (is_goal, is_better_goal) = self
                    .solution_manager
                    .check_solution(&successor, &self.id_to_chain_node);

                if is_better_goal && !better_goal_found {
                    better_goal_found = true;
                }

                if is_goal {
                    continue;
                }

                let hash_value = (self.hash_function)(successor.get_signature());
                let destination_rank = (hash_value % n_ranks) as Rank;

                if destination_rank == this_rank {
                    self.solution_manager.increment_kept();
                    let successor = N::from(successor);

                    let is_inserted = Self::open_node_inner(
                        successor,
                        &mut local_callback,
                        &mut self.registry,
                        &mut self.solution_manager,
                    );

                    if is_inserted && no_successor {
                        no_successor = false;
                    }
                } else {
                    successor.set_parent_rank(this_rank);
                    send_callback(destination_rank, successor);
                    self.solution_manager.increment_sent();

                    if no_successor {
                        no_successor = false;
                    }
                }
            }
        }

        if no_successor {
            node.get_distributed_transition_id_chain().id.set(None);
            self.id_to_chain_node.pop();
        }

        better_goal_found
    }

    pub fn finalize(
        &mut self,
        local_dual_bound: Option<T>,
    ) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        self.solution_manager.finalize(local_dual_bound)
    }
}
