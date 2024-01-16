use memoffset::offset_of;
use mpi::datatype::UserDatatype;
use mpi::traits::*;
use mpi::{Address, Rank, Tag};

use crate::distributed_id_chain::TransitionId;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PartialSolutionFixeLengthData([usize; 2], Rank, bool);

unsafe impl Equivalence for PartialSolutionFixeLengthData {
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[2, 1, 1],
            &[
                offset_of!(PartialSolutionFixeLengthData, 0) as Address,
                offset_of!(PartialSolutionFixeLengthData, 1) as Address,
                offset_of!(PartialSolutionFixeLengthData, 2) as Address,
            ],
            &[
                usize::equivalent_datatype(),
                Rank::equivalent_datatype(),
                bool::equivalent_datatype(),
            ],
        )
    }
}

pub fn send_partial_solution<C: Communicator>(
    communicator: &C,
    transition_ids: &[TransitionId],
    parent: Option<(Rank, usize)>,
    destination_rank: Rank,
    tag_fixed_length_data: Tag,
    tag_transition_ids: Tag,
) {
    let destination = communicator.process_at_rank(destination_rank);

    let fixed_length_data = PartialSolutionFixeLengthData(
        [transition_ids.len(), parent.map(|(_, id)| id).unwrap_or(0)],
        parent.map(|(rank, _)| rank).unwrap_or(0),
        parent.is_some(),
    );

    destination.buffered_send_with_tag(&fixed_length_data, tag_fixed_length_data);
    destination.buffered_send_with_tag(transition_ids, tag_transition_ids);
}

pub fn receive_partial_solution<C: Communicator>(
    communicator: &C,
    transition_ids: &mut Vec<TransitionId>,
    source_rank: Rank,
    tag_fixed_length_data: Tag,
    tag_transition_ids: Tag,
) -> Option<(Rank, usize)> {
    let source = communicator.process_at_rank(source_rank);

    let mut fixed_length_data = PartialSolutionFixeLengthData::default();
    source.receive_into_with_tag(&mut fixed_length_data, tag_fixed_length_data);

    let n = fixed_length_data.0[0];
    let offset = transition_ids.len();
    transition_ids.resize(offset + n, TransitionId::default());

    source.receive_into_with_tag(&mut transition_ids[offset..], tag_transition_ids);

    if fixed_length_data.2 {
        Some((fixed_length_data.1, fixed_length_data.0[1]))
    } else {
        None
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PartialSolutionWithTimestampFixeLengthData([usize; 3], Rank, bool);

unsafe impl Equivalence for PartialSolutionWithTimestampFixeLengthData {
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[3, 1, 1],
            &[
                offset_of!(PartialSolutionWithTimestampFixeLengthData, 0) as Address,
                offset_of!(PartialSolutionWithTimestampFixeLengthData, 1) as Address,
                offset_of!(PartialSolutionWithTimestampFixeLengthData, 2) as Address,
            ],
            &[
                usize::equivalent_datatype(),
                Rank::equivalent_datatype(),
                bool::equivalent_datatype(),
            ],
        )
    }
}

pub fn send_partial_solution_with_time_stamp<C: Communicator>(
    communicator: &C,
    transition_ids: &[TransitionId],
    parent: Option<(Rank, usize)>,
    timestamp: usize,
    destination_rank: Rank,
    tag_fixed_length_data: Tag,
    tag_transition_ids: Tag,
) {
    let destination = communicator.process_at_rank(destination_rank);

    let fixed_length_data = PartialSolutionWithTimestampFixeLengthData(
        [
            transition_ids.len(),
            parent.map(|(_, id)| id).unwrap_or(0),
            timestamp,
        ],
        parent.map(|(rank, _)| rank).unwrap_or(0),
        parent.is_some(),
    );

    destination.buffered_send_with_tag(&fixed_length_data, tag_fixed_length_data);
    destination.buffered_send_with_tag(transition_ids, tag_transition_ids);
}

pub fn receive_partial_solution_with_timestamp<C: Communicator>(
    communicator: &C,
    transition_ids: &mut Vec<TransitionId>,
    source_rank: Rank,
    timestamp: usize,
    tag_fixed_length_data: Tag,
    tag_transition_ids: Tag,
) -> (Option<(Rank, usize)>, bool) {
    let source = communicator.process_at_rank(source_rank);

    let mut fixed_length_data = PartialSolutionWithTimestampFixeLengthData::default();
    source.receive_into_with_tag(&mut fixed_length_data, tag_fixed_length_data);

    let other_timestamp = fixed_length_data.0[2];
    let n = fixed_length_data.0[0];
    let offset = transition_ids.len();
    transition_ids.resize(offset + n, TransitionId::default());

    source.receive_into_with_tag(&mut transition_ids[offset..], tag_transition_ids);

    if other_timestamp < timestamp {
        transition_ids.truncate(offset);

        return (None, false);
    }

    let parent = if fixed_length_data.2 {
        Some((fixed_length_data.1, fixed_length_data.0[1]))
    } else {
        None
    };

    (parent, true)
}
