use memoffset::offset_of;
use mpi::{datatype::UserDatatype, traits::*, Address, Rank, Tag};

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

pub struct PartialSolutionTags {
    pub fixed_length_data: Tag,
    pub transition_ids: Tag,
    pub transition_forced: Tag,
}

pub fn send_partial_solution<D: Destination>(
    destination: &D,
    transition_ids: &[usize],
    transition_forced: &[bool],
    parent: Option<(Rank, usize)>,
    tags: &PartialSolutionTags,
) {
    let fixed_length_data = PartialSolutionFixeLengthData(
        [transition_ids.len(), parent.map(|(_, id)| id).unwrap_or(0)],
        parent.map(|(rank, _)| rank).unwrap_or(0),
        parent.is_some(),
    );

    destination.buffered_send_with_tag(&fixed_length_data, tags.fixed_length_data);
    destination.buffered_send_with_tag(transition_ids, tags.transition_ids);
    destination.buffered_send_with_tag(transition_forced, tags.transition_forced);
}

pub fn receive_partial_solution<S: Source>(
    source: &S,
    transition_ids: &mut Vec<usize>,
    transition_forced: &mut Vec<bool>,
    tags: &PartialSolutionTags,
) -> Option<(Rank, usize)> {
    let mut fixed_length_data = PartialSolutionFixeLengthData::default();
    source.receive_into_with_tag(&mut fixed_length_data, tags.fixed_length_data);

    let n = fixed_length_data.0[0];
    let offset = transition_ids.len();
    transition_ids.resize(offset + n, 0);
    transition_forced.resize(offset + n, false);

    source.receive_into_with_tag(&mut transition_ids[offset..], tags.transition_ids);
    source.receive_into_with_tag(&mut transition_forced[offset..], tags.transition_forced);

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

pub fn send_partial_solution_with_time_stamp<D: Destination>(
    destination: &D,
    transition_ids: &[usize],
    transition_forced: &[bool],
    parent: Option<(Rank, usize)>,
    timestamp: usize,
    tags: &PartialSolutionTags,
) {
    let fixed_length_data = PartialSolutionWithTimestampFixeLengthData(
        [
            transition_ids.len(),
            parent.map(|(_, id)| id).unwrap_or(0),
            timestamp,
        ],
        parent.map(|(rank, _)| rank).unwrap_or(0),
        parent.is_some(),
    );

    destination.buffered_send_with_tag(&fixed_length_data, tags.fixed_length_data);
    destination.buffered_send_with_tag(transition_ids, tags.transition_ids);
    destination.buffered_send_with_tag(transition_forced, tags.transition_forced);
}

pub fn receive_partial_solution_with_timestamp<S: Source>(
    source: &S,
    transition_ids: &mut Vec<usize>,
    transition_forced: &mut Vec<bool>,
    timestamp: usize,
    tags: &PartialSolutionTags,
) -> (Option<(Rank, usize)>, bool) {
    let mut fixed_length_data = PartialSolutionWithTimestampFixeLengthData::default();
    source.receive_into_with_tag(&mut fixed_length_data, tags.fixed_length_data);

    let other_timestamp = fixed_length_data.0[2];
    let n = fixed_length_data.0[0];
    let offset = transition_ids.len();
    transition_ids.resize(offset + n, 0);
    transition_forced.resize(offset + n, false);

    source.receive_into_with_tag(&mut transition_ids[offset..], tags.transition_ids);
    source.receive_into_with_tag(&mut transition_forced[offset..], tags.transition_forced);

    if other_timestamp < timestamp {
        transition_ids.truncate(offset);
        transition_forced.truncate(offset);

        return (None, false);
    }

    let parent = if fixed_length_data.2 {
        Some((fixed_length_data.1, fixed_length_data.0[1]))
    } else {
        None
    };

    (parent, true)
}
