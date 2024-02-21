use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure::{
    exceed_bound, Beam, HashableSignatureVariables,
};
use dypdl_heuristic_search::search_algorithm::util::TimeKeeper;
use dypdl_heuristic_search::search_algorithm::{
    get_solution_cost_and_suffix, BeamSearchParameters, SearchInput, Solution, StateRegistry,
    TransitionWithId,
};
use memoffset::offset_of;
use mpi::datatype::SystemDatatype;
use mpi::{datatype::UserDatatype, topology::SimpleCommunicator, Rank, Tag};
use mpi::{traits::*, Address};
use std::fmt::Display;
use std::mem;
use std::rc::Rc;

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::distributed_id_chain::{DistributedTransitionIdChain, GetDistributedTransitionIdChain};
use crate::is_float::IsFloat;
use crate::node_communicator::NodeCommunicator;
use crate::node_data_type::NodeDatatype;
use crate::partial_solution::{
    receive_partial_solution, send_partial_solution, PartialSolutionTags,
};
use crate::statistics::Statistics;

const TAG_NODE: Tag = 0;
const TAG_ALL_NODES_SENT: Tag = 1;
const TAG_LOCAL_LAYER_MESSAGE: Tag = 2;
const TAG_PARTIAL_SOLUTION_REQUEST: Tag = 3;
const TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA: Tag = 4;
const TAG_PARTIAL_SOLUTION_TRANSITION_IDS: Tag = 5;
const TAG_PARTIAL_SOLUTION_TRANSITION_FORCED: Tag = 6;
const TAG_PARTIAL_SOLUTION: PartialSolutionTags = PartialSolutionTags {
    fixed_length_data: TAG_PARTIAL_SOLUTION_FIXED_LENGTH_DATA,
    transition_ids: TAG_PARTIAL_SOLUTION_TRANSITION_IDS,
    transition_forced: TAG_PARTIAL_SOLUTION_TRANSITION_FORCED,
};
const TAG_PARTIAL_SOLUTION_FINISHED: Tag = 7;

struct BufferedNodeCommunicator<'a, C, M, T> {
    communicator: NodeCommunicator<'a, C, M, T>,
    buffers: Vec<Vec<M>>,
    channel_open: Vec<bool>,
}

impl<'a, C, M, T> BufferedNodeCommunicator<'a, C, M, T>
where
    C: Communicator,
    M: NodeDatatype<T>,
    T: IsFloat,
{
    fn new(communicator: &'a C, tag: Tag, model: Rc<Model>, capacity: usize) -> Self {
        let inner = NodeCommunicator::new(communicator, tag, model);

        let n_ranks = communicator.size() as usize;
        let capacity = capacity / n_ranks;
        let this_rank = communicator.rank() as usize;
        let buffers = (0..n_ranks)
            .map(|rank| {
                if rank == this_rank {
                    Vec::default()
                } else {
                    Vec::with_capacity(capacity)
                }
            })
            .collect();
        let channel_open = vec![true; n_ranks];

        Self {
            communicator: inner,
            buffers,
            channel_open,
        }
    }

    fn close_channels(&mut self) {
        self.channel_open.iter_mut().for_each(|channel_open| {
            *channel_open = false;
        });
    }

    fn open_channel(&mut self, destination_rank: Rank) {
        let destination_rank = destination_rank as usize;
        self.channel_open[destination_rank] = true;
        self.buffers[destination_rank]
            .drain(..)
            .for_each(|node| self.communicator.send(destination_rank as Rank, node))
    }

    fn send(&mut self, destination_rank: Rank, node: M) {
        let destination_rank = destination_rank as usize;

        if self.channel_open[destination_rank] {
            self.communicator.send(destination_rank as Rank, node);
        } else {
            self.buffers[destination_rank].push(node);
        }
    }

    fn receive(&mut self, source_rank: Rank, primal_bound: Option<T>) -> Option<M> {
        self.communicator.receive(source_rank, primal_bound)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct LocalLayerMessage<T> {
    pruned: bool,
    is_empty: bool,
    time_out: bool,
    bound: Option<T>,
    cost: Option<T>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct LocalLayerMessageForSend<T>([bool; 5], [T; 2]);

unsafe impl<T> Equivalence for LocalLayerMessageForSend<T>
where
    T: Equivalence<Out = SystemDatatype>,
{
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[5, 2],
            &[
                offset_of!(LocalLayerMessageForSend<T>, 0) as Address,
                offset_of!(LocalLayerMessageForSend<T>, 1) as Address,
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
                message.pruned,
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
        let bound = if message.0[3] {
            Some(U::from(message.1[0]))
        } else {
            None
        };
        let cost = if message.0[4] {
            Some(U::from(message.1[1]))
        } else {
            None
        };

        Self {
            pruned: message.0[0],
            is_empty: message.0[1],
            time_out: message.0[2],
            bound,
            cost,
        }
    }
}

impl<T: IsFloat> LocalLayerMessage<T> {
    fn send<C: Communicator>(&self, communicator: &C, destination_rank: Rank, tag: Tag) {
        let destination = communicator.process_at_rank(destination_rank);

        if T::is_float() {
            let message = LocalLayerMessageForSend::<Continuous>::from(self.clone());
            destination.buffered_send_with_tag(&message, tag);
        } else {
            let message = LocalLayerMessageForSend::<Integer>::from(self.clone());
            destination.buffered_send_with_tag(&message, tag);
        }
    }

    fn receive<C: Communicator>(communicator: &C, source_rank: Rank, tag: Tag) -> Self {
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

fn retrieve_solution<C, N, V>(
    communicator: &C,
    id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
    node: &N,
    suffix: &[TransitionWithId<V>],
    forced_transitions: &[Rc<TransitionWithId<V>>],
    transitions: &[Rc<TransitionWithId<V>>],
) -> Vec<TransitionWithId<V>>
where
    C: Communicator,
    N: GetDistributedTransitionIdChain,
    V: TransitionInterface + Clone,
{
    let chain = node.get_distributed_transition_id_chain();
    let (mut transition_ids, mut transition_forced, mut parent) =
        chain.get_transition_ids_in_this_rank(id_to_chain_node);

    while let Some((parent_rank, parent_id)) = parent {
        if parent_rank == communicator.rank() {
            let chain = &id_to_chain_node[parent_id];
            let (tmp_transition_ids, tmp_transition_forced, tmp_parent) =
                chain.get_transition_ids_in_this_rank(id_to_chain_node);
            transition_ids.extend(tmp_transition_ids);
            transition_forced.extend(tmp_transition_forced);
            parent = tmp_parent;
        } else {
            let destination_process = communicator.process_at_rank(parent_rank);
            destination_process.buffered_send_with_tag(&parent_id, TAG_PARTIAL_SOLUTION_REQUEST);

            parent = receive_partial_solution(
                &destination_process,
                &mut transition_ids,
                &mut transition_forced,
                &TAG_PARTIAL_SOLUTION,
            );
        }
    }

    for destination_rank in 0..communicator.size() {
        if destination_rank != communicator.rank() {
            let buf: [u8; 0] = [];
            let destination_process = communicator.process_at_rank(destination_rank);
            destination_process.buffered_send_with_tag(&buf, TAG_PARTIAL_SOLUTION_FINISHED);
        }
    }

    let mut solution = transition_ids
        .iter()
        .zip(transition_forced.iter())
        .rev()
        .map(|(id, forced)| {
            if *forced {
                forced_transitions[*id].as_ref().clone()
            } else {
                transitions[*id].as_ref().clone()
            }
        })
        .collect::<Vec<_>>();

    solution.extend_from_slice(suffix);

    solution
}

fn wait_retrieve_solution<C>(
    communicator: &C,
    id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
) where
    C: Communicator,
{
    loop {
        let any_process = communicator.any_process();

        if let Some(status) = any_process.immediate_probe_with_tag(TAG_PARTIAL_SOLUTION_REQUEST) {
            let rank = status.source_rank();
            let mut chain_id = 0;
            let source_process = communicator.process_at_rank(rank);
            source_process.receive_into_with_tag(&mut chain_id, TAG_PARTIAL_SOLUTION_REQUEST);

            let chain = &id_to_chain_node[chain_id];
            let (transition_ids, transition_forced, parent) =
                chain.get_transition_ids_in_this_rank(id_to_chain_node);
            send_partial_solution(
                &source_process,
                &transition_ids,
                &transition_forced,
                parent,
                &TAG_PARTIAL_SOLUTION,
            )
        }

        if let Some(status) = communicator
            .any_process()
            .immediate_probe_with_tag(TAG_PARTIAL_SOLUTION_FINISHED)
        {
            let mut buf: [u8; 0] = [];
            let source_process = communicator.process_at_rank(status.source_rank());
            source_process.receive_into_with_tag(&mut buf, TAG_PARTIAL_SOLUTION_FINISHED);
            return;
        }
    }
}

pub fn hd_beam_search2<'a, T, N, M, E, B, F, V>(
    input: &'a SearchInput<'a, M, TransitionWithId<V>>,
    transition_evaluator: E,
    base_cost_evaluator: B,
    parameters: BeamSearchParameters<T>,
    hash_function: F,
    communicator: &'a SimpleCommunicator,
    controller_rank: Rank,
) -> (Solution<T, TransitionWithId<V>>, Option<Rank>, Statistics)
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    M: Clone + NodeDatatype<T>,
    E: Fn(&N, &TransitionWithId<V>, Option<T>) -> Option<M>,
    B: Fn(T, T) -> T,
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
    let mut current_beam = Beam::<_, N>::new(parameters.beam_size);
    let mut next_beam = Beam::<_, N>::new(parameters.beam_size);
    let mut registry = StateRegistry::<_, N>::new(model.clone());
    let capacity = parameters
        .parameters
        .initial_registry_capacity
        .unwrap_or_else(|| current_beam.capacity());
    registry.reserve(capacity);

    let mut sent = 0;
    let mut kept = 0;
    let mut received = 0;
    let mut generated = 0;

    if let Some(node) = input.node.clone() {
        let hash_value = hash_function(node.get_signature());
        let assigned_rank = (hash_value % n_ranks) as Rank;

        if assigned_rank == this_rank {
            let node = N::from(node);
            current_beam.insert(&mut registry, node);
            generated += 1;

            if !parameters.keep_all_layers {
                registry.clear();
            }
        }
    }

    let mut expanded = 0;
    let mut layer_index = 0;

    let mut pruned = false;
    let mut best_dual_bound = None;
    let mut layer_dual_bound = None;
    let mut removed_dual_bound = None;
    let mut time_out = this_rank == controller_rank && time_keeper.check_time_limit(quiet);
    let mut incumbent = None;

    for destination_rank in 0..n_ranks as Rank {
        if destination_rank != this_rank {
            let message = LocalLayerMessage::<T> {
                pruned,
                is_empty: current_beam.is_empty(),
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
        let mut is_empty = current_beam.is_empty();
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
            layer_dual_bound = removed_dual_bound;

            let mut opened = 0;
            let mut sent_all = false;
            let mut expanded_all = false;
            let mut received_all = 0;
            let mut best_dual_bound_checked = false;

            let mut iter = if parameters.keep_all_layers {
                current_beam.close_and_drain()
            } else {
                current_beam.drain()
            };

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

                        pruned |= information.pruned;
                        is_empty &= information.is_empty;
                        time_out |= information.time_out;

                        if let Some(bound) = information.bound {
                            if !exceed_bound(model, bound, previous_layer_dual_bound) {
                                previous_layer_dual_bound = Some(bound);
                            }
                        }

                        if let Some(other) = information.cost {
                            if !exceed_bound(model, other, cost)
                                || (Some(other) == cost && source_rank < goal_rank.unwrap())
                            {
                                cost = Some(other);
                                goal_rank = Some(source_rank);

                                if !exceed_bound(model, other, primal_bound) {
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
                        if exceed_bound(model, value, primal_bound) {
                            best_dual_bound = primal_bound;
                        } else if best_dual_bound
                            .map_or(true, |bound| !exceed_bound(model, bound, Some(value)))
                        {
                            best_dual_bound = Some(value);
                        }
                    }

                    best_dual_bound_checked = true;
                }

                if !expanded_all {
                    // Expands a node.
                    if let Some(node) = iter.next() {
                        if let Some(bound) = node.bound(model) {
                            if exceed_bound(model, bound, primal_bound) {
                                continue;
                            }
                        }

                        if let Some((cost, suffix)) = get_solution_cost_and_suffix(
                            model,
                            &*node,
                            suffix,
                            &base_cost_evaluator,
                        ) {
                            if !exceed_bound(model, cost, primal_bound) {
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

                        expanded += 1;

                        let mut no_successor = true;
                        node.get_distributed_transition_id_chain()
                            .id
                            .set(Some(id_to_chain_node.len()));

                        for transition in generator.applicable_transitions(node.state()) {
                            if let Some(successor) =
                                transition_evaluator(&node, &transition, primal_bound)
                            {
                                let hash_value = hash_function(successor.get_signature());
                                let destination_rank = (hash_value % n_ranks) as Rank;

                                if destination_rank == this_rank {
                                    kept += 1;
                                    let successor = N::from(successor);
                                    let successor_bound = successor.bound(model);
                                    let status = next_beam.insert(&mut registry, successor);

                                    if !pruned && (status.is_pruned || status.removed.is_some()) {
                                        pruned = true;
                                    }

                                    if let Some(bound) = successor_bound {
                                        if !exceed_bound(model, bound, layer_dual_bound) {
                                            layer_dual_bound = Some(bound);
                                        }

                                        if status.is_pruned
                                            && !exceed_bound(model, bound, removed_dual_bound)
                                        {
                                            removed_dual_bound = Some(bound);
                                        }
                                    }

                                    if let Some(bound) =
                                        status.removed.and_then(|removed| removed.bound(model))
                                    {
                                        if !exceed_bound(model, bound, removed_dual_bound) {
                                            removed_dual_bound = Some(bound);
                                        }
                                    }

                                    if status.is_newly_registered {
                                        generated += 1;
                                    }

                                    if no_successor && status.is_inserted {
                                        no_successor = false;
                                    }
                                } else {
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
                            node.get_distributed_transition_id_chain().id.set(None);
                        } else {
                            id_to_chain_node
                                .push(node.get_distributed_transition_id_chain().clone());
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
                        let source_rank = status.source_rank();
                        received += 1;
                        source_to_counter[source_rank as usize] += 1;

                        if source_to_counter[source_rank as usize] == 0 {
                            received_all += 1;
                        }

                        if let Some(node) = node_communicator.receive(source_rank, primal_bound) {
                            let node = N::from(node);
                            let node_bound = node.bound(model);
                            let status = next_beam.insert(&mut registry, node);

                            if !pruned && (status.is_pruned || status.removed.is_some()) {
                                pruned = true;
                            }

                            if let Some(bound) = node_bound {
                                if !exceed_bound(model, bound, layer_dual_bound) {
                                    layer_dual_bound = Some(bound);
                                }

                                if status.is_pruned
                                    && !exceed_bound(model, bound, removed_dual_bound)
                                {
                                    removed_dual_bound = Some(bound);
                                }
                            }

                            if let Some(bound) =
                                status.removed.and_then(|removed| removed.bound(model))
                            {
                                if !exceed_bound(model, bound, removed_dual_bound) {
                                    removed_dual_bound = Some(bound);
                                }
                            }

                            if status.is_newly_registered {
                                generated += 1;
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
            let proved = !pruned && is_empty;

            let mut solution = Solution {
                cost,
                best_bound: best_dual_bound,
                is_optimal: cost.is_some() && proved,
                is_infeasible: cost.is_none() && proved,
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
            };

            // Found a solution.
            if let Some(goal_rank) = goal_rank {
                if cost == best_dual_bound {
                    solution.is_optimal = true;
                }

                if goal_rank == this_rank {
                    let (node, cost, suffix) = incumbent.unwrap();
                    solution.transitions = retrieve_solution(
                        communicator,
                        &id_to_chain_node,
                        node.as_ref(),
                        suffix,
                        &generator.forced_transitions,
                        &generator.transitions,
                    );
                    solution.cost = Some(cost);
                } else {
                    wait_retrieve_solution(communicator, &id_to_chain_node);
                }
            }

            return (solution, goal_rank, statistics);
        }

        node_communicator.close_channels();

        time_out = this_rank == 0 && time_keeper.check_time_limit(quiet);

        let local_layer_message = LocalLayerMessage::<T> {
            pruned,
            is_empty: next_beam.is_empty(),
            time_out,
            bound: layer_dual_bound,
            cost: incumbent.as_ref().map(|(_, cost, _)| *cost),
        };

        for destination_rank in 0..n_ranks as Rank {
            if destination_rank != this_rank {
                local_layer_message.send(communicator, destination_rank, TAG_LOCAL_LAYER_MESSAGE);
            }
        }

        mem::swap(&mut current_beam, &mut next_beam);

        if !parameters.keep_all_layers {
            registry.clear();
        }

        layer_index += 1;
    }
}
