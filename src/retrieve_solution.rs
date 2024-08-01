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
            )
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
