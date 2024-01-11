use mpi::traits::*;
use mpi::{Rank, Tag};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartialSolutionTags {
    pub tag_n: Tag,
    pub tag_ids: Tag,
    pub tag_forced: Tag,
    pub tag_has_parent: Tag,
    pub tag_parent_rank: Tag,
}

pub fn send_partial_solution<C: Communicator>(
    communicator: &C,
    transition_ids: &[usize],
    forced: &[bool],
    parent_rank: Option<Rank>,
    destination_rank: Rank,
    tags: &PartialSolutionTags,
) {
    let destination = communicator.process_at_rank(destination_rank);

    let n = transition_ids.len();
    destination.send_with_tag(&n, tags.tag_n);
    destination.send_with_tag(transition_ids, tags.tag_ids);
    destination.send_with_tag(forced, tags.tag_forced);

    if let Some(parent_rank) = parent_rank {
        destination.send_with_tag(&true, tags.tag_has_parent);
        destination.send_with_tag(&parent_rank, tags.tag_parent_rank);
    } else {
        destination.send_with_tag(&false, tags.tag_has_parent);
    }
}

pub fn receive_partial_solution<C: Communicator>(
    communicator: &C,
    transition_ids: &mut Vec<usize>,
    forced: &mut Vec<bool>,
    source_rank: Rank,
    tags: &PartialSolutionTags,
) -> Option<Rank> {
    debug_assert_eq!(transition_ids.len(), forced.len());

    let source = communicator.process_at_rank(source_rank);

    let mut n = 0;
    source.receive_into_with_tag(&mut n, tags.tag_n);
    let offset = transition_ids.len();
    transition_ids.resize(offset + n, 0);
    forced.resize(offset + n, false);

    source.receive_into_with_tag(&mut transition_ids[offset..], tags.tag_ids);
    source.receive_into_with_tag(&mut forced[offset..], tags.tag_forced);

    let mut has_parent = false;
    source.receive_into_with_tag(&mut has_parent, tags.tag_has_parent);

    if has_parent {
        let mut parent_rank = 0;
        source.receive_into_with_tag(&mut parent_rank, tags.tag_parent_rank);
        Some(parent_rank)
    } else {
        None
    }
}
