use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::{
    search_algorithm::{
        self,
        data_structure::{self, HashableSignatureVariables, StateWithHashableSignatureVariables},
        util::TimeKeeper,
        SearchInput, Solution, StateInRegistry, StateRegistry, TransitionWithId,
    },
    Parameters,
};
use mpi::{
    datatype::{SystemDatatype, UserDatatype},
    topology::SimpleCommunicator,
    traits::*,
    Address, Rank, Tag,
};
use std::{collections::VecDeque, fmt::Display};
use std::{mem, rc::Rc};

use crate::is_float::IsFloat;
use crate::node_message::NodeMessage;
use crate::partial_solution::PartialSolutionTags;
use crate::retrieve_solution::{self, RetrieveSolutionTags};
use crate::statistics::Statistics;
use crate::{
    bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain,
    distributed_id_chain::DistributedTransitionIdChain, NodeGenerationResult,
};
use crate::{hd_beam_search2::BufferedNodeCommunicator, MpiAnytimeSearchEvaluators};

const TAG_NODE: Tag = 0;
const TAG_ALL_NODES_SENT: Tag = 1;
const TAG_LOCAL_LAYER_MESSAGE: Tag = 2;
const TAG_PARTIAL_SOLUTION_REQUEST: Tag = 3;
const TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA: Tag = 4;
const TAG_PARTIAL_SOLUTION_TRANSITION_IDS: Tag = 5;
const TAG_PARTIAL_SOLUTION_TRANSITION_FORCED: Tag = 6;
const TAG_PARTIAL_SOLUTION_FINISHED: Tag = 7;
const TAG_RETRIEVE_SOLUTION: RetrieveSolutionTags = RetrieveSolutionTags {
    tag_partial_solution_request: TAG_PARTIAL_SOLUTION_REQUEST,
    tag_partial_solution: PartialSolutionTags {
        fixed_length_data: TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA,
        transition_ids: TAG_PARTIAL_SOLUTION_TRANSITION_IDS,
        transition_forced: TAG_PARTIAL_SOLUTION_TRANSITION_FORCED,
    },
    tag_partial_solution_finished: TAG_PARTIAL_SOLUTION_FINISHED,
};

#[derive(Clone, Debug, Default, PartialEq)]
struct LocalLayerMessage<T> {
    is_empty: bool,
    time_out: bool,
    bound: Option<T>,
    cost: Option<T>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct LocalLayerMessageForSend<T>([bool; 4], [T; 2]);

unsafe impl<T> Equivalence for LocalLayerMessageForSend<T>
where
    T: Equivalence<Out = SystemDatatype>,
{
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[4, 2],
            &[
                memoffset::offset_of!(LocalLayerMessageForSend<T>, 0) as Address,
                memoffset::offset_of!(LocalLayerMessageForSend<T>, 1) as Address,
            ],
            &[bool::equivalent_datatype(), T::equivalent_datatype()],
        )
    }
}

impl<T, U> From<LocalLayerMessage<T>> for LocalLayerMessageForSend<U>
where
    T: Numeric,
    U: Numeric,
{
    fn from(message: LocalLayerMessage<T>) -> Self {
        let bound = message
            .bound
            .map_or_else(|| U::default(), |bound| U::from(bound));
        let cost = message
            .cost
            .map_or_else(|| U::default(), |cost| U::from(cost));

        Self(
            [
                message.is_empty,
                message.time_out,
                message.bound.is_some(),
                message.cost.is_some(),
            ],
            [bound, cost],
        )
    }
}

impl<T, U> From<LocalLayerMessageForSend<T>> for LocalLayerMessage<U>
where
    T: Numeric,
    U: Numeric,
{
    fn from(message: LocalLayerMessageForSend<T>) -> Self {
        let bound = if message.0[2] {
            Some(U::from(message.1[0]))
        } else {
            None
        };
        let cost = if message.0[3] {
            Some(U::from(message.1[1]))
        } else {
            None
        };

        Self {
            is_empty: message.0[0],
            time_out: message.0[1],
            bound,
            cost,
        }
    }
}

impl<T: IsFloat> LocalLayerMessage<T> {
    pub fn send<C: Communicator>(&self, communicator: &C, destination_rank: Rank, tag: Tag) {
        let destination = communicator.process_at_rank(destination_rank);

        if T::is_float() {
            let message = LocalLayerMessageForSend::<Continuous>::from(self.clone());
            destination.buffered_send_with_tag(&message, tag);
        } else {
            let message = LocalLayerMessageForSend::<Integer>::from(self.clone());
            destination.buffered_send_with_tag(&message, tag);
        }
    }

    pub fn receive<C: Communicator>(communicator: &C, source_rank: Rank, tag: Tag) -> Self {
        let source = communicator.process_at_rank(source_rank);

        if T::is_float() {
            let mut message = LocalLayerMessageForSend::<Continuous>::default();
            source.receive_into_with_tag(&mut message, tag);
            Self::from(message)
        } else {
            let mut message = LocalLayerMessageForSend::<Integer>::default();
            source.receive_into_with_tag(&mut message, tag);
            Self::from(message)
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Hdbrfs2Parameters<T> {
    pub parameters: Parameters<T>,
    pub dual_bound: Option<T>,
}

pub fn hd_breadth_first_search2<'a, T, N, M, L, R, B, F, V>(
    input: &'a SearchInput<'a, M, TransitionWithId<V>>,
    evaluators: MpiAnytimeSearchEvaluators<L, R, B>,
    parameters: Hdbrfs2Parameters<T>,
    hash_function: F,
    communicator: &'a SimpleCommunicator,
    controller_rank: Rank,
) -> (Solution<T, TransitionWithId<V>>, Option<Rank>, Statistics)
where
    T: Numeric + IsFloat + Ord + Display,
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
    F: Fn(&HashableSignatureVariables) -> u64,
    V: TransitionInterface + Clone + Default,
{
    let this_rank = communicator.rank();
    let time_keeper = parameters
        .parameters
        .time_limit
        .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);

    let quiet = this_rank != 0 || parameters.parameters.quiet;
    let mut primal_bound = parameters.parameters.primal_bound;
    let n_ranks = communicator.size() as u64;

    let model = &input.generator.model;
    let generator = &input.generator;
    let suffix = input.solution_suffix;
    let mut current_open = VecDeque::default();
    let mut next_open = VecDeque::default();
    let mut registry = StateRegistry::<_, N>::new(model.clone());
    let capacity = parameters
        .parameters
        .initial_registry_capacity
        .unwrap_or_else(|| current_open.capacity());
    registry.reserve(capacity);

    let mut sent = 0;
    let mut kept = 0;
    let mut received = 0;
    let mut generated = 0;

    if let Some(node) = input.node.clone() {
        let hash_value = hash_function(node.signature());
        let assigned_rank = (hash_value % n_ranks) as Rank;

        if assigned_rank == this_rank {
            let node = N::from(node);
            let result = registry.insert(node);

            if let Some(node) = result.information {
                current_open.push_back(node);
                generated += 1;
            }

            registry.clear();
        }
    }

    let mut local_successor_evaluator = evaluators.local_successor_evaluator;
    let mut remote_successor_evaluator = evaluators.remote_successor_evaluator;
    let mut base_cost_evaluator = evaluators.base_cost_evaluator;

    let mut expanded = 0;
    let mut dominated_before_closed = 0;
    let mut first_expanded_timestamp = 0.0;
    let mut last_expanded_timestamp = 0.0;
    let mut first_received_timestamp = 0.0;
    let mut last_received_timestamp = 0.0;
    let mut layer_index = 0;

    let mut best_dual_bound = parameters.dual_bound;
    let mut layer_dual_bound = None;
    let mut time_out = this_rank == controller_rank && time_keeper.check_time_limit(quiet);
    let mut incumbent = None;

    for destination_rank in 0..n_ranks as Rank {
        if destination_rank != this_rank {
            let message = LocalLayerMessage::<T> {
                is_empty: current_open.is_empty(),
                time_out,
                bound: layer_dual_bound,
                cost: None,
            };
            message.send(communicator, destination_rank, TAG_LOCAL_LAYER_MESSAGE);
        }
    }

    let mut id_to_chain_node = vec![];
    let mut node_communicator =
        BufferedNodeCommunicator::<_, M, T>::new(communicator, TAG_NODE, model.clone(), capacity);
    let any_process = communicator.any_process();
    let mut destination_to_n_sent = vec![0; n_ranks as usize];
    let mut source_to_counter = vec![0; n_ranks as usize];

    loop {
        let mut is_empty = current_open.is_empty();
        let mut cost = incumbent.as_ref().map(|(_, cost, _)| *cost);
        let mut goal_rank = if cost.is_some() {
            Some(this_rank)
        } else {
            None
        };
        destination_to_n_sent.iter_mut().for_each(|n_sent| {
            *n_sent = 0;
        });
        source_to_counter.iter_mut().for_each(|counter| {
            *counter = 0;
        });

        {
            let mut previous_layer_dual_bound = layer_dual_bound;
            layer_dual_bound = None;

            let mut opened = 0;
            let mut sent_all = false;
            let mut expanded_all = false;
            let mut received_all = 0;
            let mut best_dual_bound_checked = false;

            let mut iter = current_open.drain(..);

            while !sent_all || received_all < n_ranks - 1 {
                if opened < n_ranks - 1 {
                    // Receives the information of the previous layer.
                    while let Some(status) =
                        any_process.immediate_probe_with_tag(TAG_LOCAL_LAYER_MESSAGE)
                    {
                        let source_rank = status.source_rank();
                        let information = LocalLayerMessage::<T>::receive(
                            communicator,
                            source_rank,
                            TAG_LOCAL_LAYER_MESSAGE,
                        );
                        // Now, we are sure that the sender thread finishes the previous layer.
                        // We can start to send nodes to the thread.
                        node_communicator.open_channel(source_rank);
                        opened += 1;

                        is_empty &= information.is_empty;
                        time_out |= information.time_out;

                        if let Some(bound) = information.bound {
                            if !data_structure::exceed_bound(
                                model,
                                bound,
                                previous_layer_dual_bound,
                            ) {
                                previous_layer_dual_bound = Some(bound);
                            }
                        }

                        if let Some(other) = information.cost {
                            if !data_structure::exceed_bound(model, other, cost)
                                || (Some(other) == cost && source_rank < goal_rank.unwrap())
                            {
                                cost = Some(other);
                                goal_rank = Some(source_rank);

                                if !data_structure::exceed_bound(model, other, primal_bound) {
                                    primal_bound = Some(other);
                                }
                            }
                        }

                        if opened == n_ranks - 1 {
                            break;
                        }
                    }
                }

                if !best_dual_bound_checked && opened == n_ranks - 1 {
                    if let Some(value) = previous_layer_dual_bound {
                        if data_structure::exceed_bound(model, value, primal_bound) {
                            best_dual_bound = primal_bound;
                        } else if best_dual_bound.map_or(true, |bound| {
                            !data_structure::exceed_bound(model, bound, Some(value))
                        }) {
                            best_dual_bound = Some(value);
                        }
                    }

                    best_dual_bound_checked = true;
                }

                if !expanded_all {
                    // Expands a node.
                    if let Some(node) = iter.next() {
                        if let Some(bound) = node.bound(model) {
                            if data_structure::exceed_bound(model, bound, primal_bound) {
                                continue;
                            }
                        }

                        if let Some((cost, suffix)) = search_algorithm::get_solution_cost_and_suffix(
                            model,
                            &*node,
                            suffix,
                            &mut base_cost_evaluator,
                        ) {
                            if !data_structure::exceed_bound(model, cost, primal_bound) {
                                primal_bound = Some(cost);
                                incumbent = Some((node, cost, suffix));

                                // Optimal solution, ignore remaining open nodes.
                                if Some(cost) == best_dual_bound {
                                    expanded_all = true;
                                }
                            }
                            continue;
                        }

                        if time_out {
                            continue;
                        }

                        last_expanded_timestamp = time_keeper.elapsed_time();

                        if expanded == 0 {
                            first_expanded_timestamp = last_expanded_timestamp;
                        }

                        expanded += 1;

                        let mut no_successor = true;
                        node.get_rc_distributed_transition_id_chain()
                            .id
                            .set(Some(id_to_chain_node.len()));

                        for transition in generator.applicable_transitions(node.state()) {
                            if let Some((successor_state, g)) = model.generate_successor_state::<_, StateWithHashableSignatureVariables, _, _>(
                                node.state(),
                                node.cost(model),
                                transition.as_ref(),
                                None,
                            ) {
                                let hash_value =
                                    hash_function(&successor_state.signature_variables);
                                let destination_rank = (hash_value % n_ranks) as Rank;

                                if destination_rank == this_rank {
                                    let successor_state = StateInRegistry::from(successor_state);
                                    let result = local_successor_evaluator(
                                        successor_state,
                                        g,
                                        &transition,
                                        node.get_distributed_transition_id_chain(),
                                        &mut registry,
                                        primal_bound,
                                    );

                                    if !result.is_pruned_by_bound {
                                        kept += 1;
                                    }

                                    if result.dominated_before_closed == 0 && result.dominated_after_closed == 0
                                    {
                                        generated += 1;
                                    }

                                    if let Some(successor) = result.node {
                                        if let Some(bound) = successor.bound(model) {
                                            if !data_structure::exceed_bound(
                                                model,
                                                bound,
                                                layer_dual_bound,
                                            ) {
                                                layer_dual_bound = Some(bound);
                                            }
                                        }

                                        next_open.push_back(successor);
                                        no_successor = false;
                                    }
                                } else if let Some(successor) = remote_successor_evaluator(
                                    successor_state,
                                    g,
                                    &transition,
                                    node.get_distributed_transition_id_chain(),
                                    primal_bound,
                                ) {
                                    successor.set_parent_rank(this_rank);
                                    node_communicator.send(destination_rank, successor);
                                    sent += 1;
                                    destination_to_n_sent[destination_rank as usize] += 1;

                                    if no_successor {
                                        no_successor = false;
                                    }
                                }
                            }
                        }

                        if no_successor {
                            node.get_rc_distributed_transition_id_chain().id.set(None);
                        } else {
                            id_to_chain_node
                                .push(node.get_rc_distributed_transition_id_chain().clone());
                        }
                    } else {
                        expanded_all = true;
                    }
                }

                if expanded_all && opened == n_ranks - 1 && !sent_all {
                    sent_all = true;

                    for destination_rank in 0..n_ranks as Rank {
                        if destination_rank != this_rank {
                            let destination_process =
                                communicator.process_at_rank(destination_rank);
                            destination_process.buffered_send_with_tag(
                                &destination_to_n_sent[destination_rank as usize],
                                TAG_ALL_NODES_SENT,
                            );
                        }
                    }
                }

                if received_all < n_ranks - 1 {
                    // Receives a node.
                    while let Some(status) = any_process.immediate_probe_with_tag(TAG_NODE) {
                        last_received_timestamp = time_keeper.elapsed_time();

                        if received == 0 {
                            first_received_timestamp = last_received_timestamp;
                        }

                        received += 1;
                        let source_rank = status.source_rank();
                        source_to_counter[source_rank as usize] += 1;

                        if source_to_counter[source_rank as usize] == 0 {
                            received_all += 1;
                        }

                        if let Some(node) = node_communicator.receive(source_rank, primal_bound) {
                            let node = N::from(node);
                            let result = registry.insert(node);

                            for d in result.dominated.iter() {
                                if !d.is_closed() {
                                    d.close();
                                    dominated_before_closed += 1;
                                }
                            }

                            if result.dominated.is_empty() {
                                generated += 1;
                            }

                            if let Some(node) = result.information {
                                if let Some(bound) = node.bound(model) {
                                    if !data_structure::exceed_bound(model, bound, layer_dual_bound)
                                    {
                                        layer_dual_bound = Some(bound);
                                    }
                                }

                                next_open.push_back(node);
                            }
                        }

                        if received_all == n_ranks - 1 {
                            break;
                        }
                    }
                }

                if received_all < n_ranks - 1 {
                    // Receives a special message.
                    while let Some(status) =
                        any_process.immediate_probe_with_tag(TAG_ALL_NODES_SENT)
                    {
                        let source_rank = status.source_rank();
                        let mut total_sent: i32 = 0;
                        let source_process = communicator.process_at_rank(source_rank);
                        source_process.receive_into_with_tag(&mut total_sent, TAG_ALL_NODES_SENT);
                        let source_rank = source_rank as usize;
                        source_to_counter[source_rank] -= total_sent;

                        if source_to_counter[source_rank] == 0 {
                            received_all += 1;

                            if received_all == n_ranks - 1 {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if !quiet {
            println!(
                "Searched layer: {}, elapsed time: {}",
                layer_index,
                time_keeper.elapsed_time()
            );
        }

        // Checks if the search is finished.
        if is_empty || time_out || goal_rank.is_some() {
            let mut solution = Solution {
                cost,
                best_bound: best_dual_bound,
                is_optimal: cost.is_some() && is_empty,
                is_infeasible: cost.is_none() && is_empty,
                transitions: vec![],
                expanded,
                generated,
                time: time_keeper.elapsed_time(),
                time_out,
            };
            let statistics = Statistics {
                expanded,
                generated,
                sent,
                kept,
                received,
                dominated_before_closed,
                dominated_after_closed: 0,
                first_expanded_timestamp,
                last_expanded_timestamp,
                first_received_timestamp,
                last_received_timestamp,
            };

            // Found a solution.
            if let Some(goal_rank) = goal_rank {
                if cost == best_dual_bound {
                    solution.is_optimal = true;
                }

                if goal_rank == this_rank {
                    let (node, cost, suffix) = incumbent.unwrap();
                    solution.transitions = retrieve_solution::retrieve_solution(
                        communicator,
                        &id_to_chain_node,
                        node.as_ref(),
                        suffix,
                        &generator.forced_transitions,
                        &generator.transitions,
                        &TAG_RETRIEVE_SOLUTION,
                    );
                    solution.cost = Some(cost);
                } else {
                    retrieve_solution::wait_retrieve_solution(
                        communicator,
                        &id_to_chain_node,
                        &TAG_RETRIEVE_SOLUTION,
                    );
                }
            }

            // Wait for the other ranks to finish.
            communicator.barrier();

            return (solution, goal_rank, statistics);
        }

        node_communicator.close_channels();

        time_out = this_rank == 0 && time_keeper.check_time_limit(quiet);

        let local_layer_message = LocalLayerMessage::<T> {
            is_empty: next_open.is_empty(),
            time_out,
            bound: layer_dual_bound,
            cost: incumbent.as_ref().map(|(_, cost, _)| *cost),
        };

        for destination_rank in 0..n_ranks as Rank {
            if destination_rank != this_rank {
                local_layer_message.send(communicator, destination_rank, TAG_LOCAL_LAYER_MESSAGE);
            }
        }

        mem::swap(&mut current_open, &mut next_open);
        registry.clear();

        layer_index += 1;
    }
}
