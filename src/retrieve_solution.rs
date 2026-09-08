use dypdl::prelude::*;
use dypdl_heuristic_search::search_algorithm::TransitionWithId;
use mpi::{traits::*, Tag};
use std::rc::Rc;

use crate::distributed_id_chain::{DistributedTransitionIdChain, GeRcDistributedTransitionIdChain};
use crate::partial_solution::{
    receive_partial_solution, send_partial_solution, PartialSolutionTags,
};

pub struct RetrieveSolutionTags {
    pub tag_partial_solution_request: Tag,
    pub tag_partial_solution: PartialSolutionTags,
    pub tag_partial_solution_finished: Tag,
}

pub fn retrieve_solution<C, N, V>(
    communicator: &C,
    id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
    node: &N,
    suffix: &[TransitionWithId<V>],
    forced_transitions: &[Rc<TransitionWithId<V>>],
    transitions: &[Rc<TransitionWithId<V>>],
    tags: &RetrieveSolutionTags,
) -> Vec<TransitionWithId<V>>
where
    C: Communicator,
    N: GeRcDistributedTransitionIdChain,
    V: TransitionInterface + Clone,
{
    retrieve_solution_and_count_control_messages(
        communicator,
        id_to_chain_node,
        node,
        suffix,
        forced_transitions,
        transitions,
        tags,
        |_| {},
    )
}

#[allow(clippy::too_many_arguments)]
pub fn retrieve_solution_and_count_control_messages<C, N, V, F>(
    communicator: &C,
    id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
    node: &N,
    suffix: &[TransitionWithId<V>],
    forced_transitions: &[Rc<TransitionWithId<V>>],
    transitions: &[Rc<TransitionWithId<V>>],
    tags: &RetrieveSolutionTags,
    mut count_control_message: F,
) -> Vec<TransitionWithId<V>>
where
    C: Communicator,
    N: GeRcDistributedTransitionIdChain,
    V: TransitionInterface + Clone,
    F: FnMut(Tag),
{
    let chain = node.get_rc_distributed_transition_id_chain();
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
            destination_process
                .buffered_send_with_tag(&parent_id, tags.tag_partial_solution_request);
            count_control_message(tags.tag_partial_solution_request);

            parent = receive_partial_solution(
                &destination_process,
                &mut transition_ids,
                &mut transition_forced,
                &tags.tag_partial_solution,
            );
        }
    }

    for destination_rank in 0..communicator.size() {
        if destination_rank != communicator.rank() {
            let buf: [u8; 0] = [];
            let destination_process = communicator.process_at_rank(destination_rank);
            destination_process.buffered_send_with_tag(&buf, tags.tag_partial_solution_finished);
            count_control_message(tags.tag_partial_solution_finished);
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

pub fn wait_retrieve_solution<C>(
    communicator: &C,
    id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
    tags: &RetrieveSolutionTags,
) where
    C: Communicator,
{
    wait_retrieve_solution_and_count_control_messages(communicator, id_to_chain_node, tags, |_| {})
}

pub fn wait_retrieve_solution_and_count_control_messages<C, F>(
    communicator: &C,
    id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
    tags: &RetrieveSolutionTags,
    mut count_control_message: F,
) where
    C: Communicator,
    F: FnMut(Tag),
{
    loop {
        let any_process = communicator.any_process();

        if let Some(status) =
            any_process.immediate_probe_with_tag(tags.tag_partial_solution_request)
        {
            let rank = status.source_rank();
            let mut chain_id = 0;
            let source_process = communicator.process_at_rank(rank);
            source_process.receive_into_with_tag(&mut chain_id, tags.tag_partial_solution_request);

            let chain = &id_to_chain_node[chain_id];
            let (transition_ids, transition_forced, parent) =
                chain.get_transition_ids_in_this_rank(id_to_chain_node);
            send_partial_solution(
                &source_process,
                &transition_ids,
                &transition_forced,
                parent,
                &tags.tag_partial_solution,
            );
            count_control_message(tags.tag_partial_solution.fixed_length_data);
            count_control_message(tags.tag_partial_solution.transition_ids);
            count_control_message(tags.tag_partial_solution.transition_forced);
        }

        if let Some(status) = communicator
            .any_process()
            .immediate_probe_with_tag(tags.tag_partial_solution_finished)
        {
            let mut buf: [u8; 0] = [];
            let source_process = communicator.process_at_rank(status.source_rank());
            source_process.receive_into_with_tag(&mut buf, tags.tag_partial_solution_finished);
            return;
        }
    }
}
