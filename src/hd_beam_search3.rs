use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::HashableSignatureVariables, util::TimeKeeper, BeamSearchParameters,
    SearchInput, Solution, TransitionWithId,
};
use mpi::{topology::SimpleCommunicator, traits::*, Rank, Tag};
use std::fmt::Display;

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::is_float::IsFloat;
use crate::layered_beams::LayeredBeams;
use crate::local_layer_message::LocalLayerMessage;
use crate::node_communicator::NodeDepthCommunicator;
use crate::node_message::NodeMessage;
use crate::partial_solution::PartialSolutionTags;
use crate::retrieve_solution::{self, RetrieveSolutionTags};
use crate::statistics::Statistics;

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

pub fn hd_beam_search3<'a, T, N, M, E, B, F, V>(
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
    M: Clone + NodeMessage<T>,
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

    let mut layered_beams = LayeredBeams::new(model.clone(), parameters.beam_size);

    let mut sent = 0;
    let mut kept = 0;
    let mut received = 0;
    let mut generated = 0;

    if let Some(node) = input.node.clone() {
        let hash_value = hash_function(node.signature());
        let assigned_rank = (hash_value % n_ranks) as Rank;

        if assigned_rank == this_rank {
            let node = N::from(node);
            layered_beams.insert(node, 0);
        }
    }

    let mut expanded = 0;
    let mut first_expanded_timestamp = 0.0;
    let mut last_expanded_timestamp = 0.0;
    let mut first_received_timestamp = 0.0;
    let mut last_received_timestamp = 0.0;

    let mut pruned = false;
    let mut time_out = this_rank == controller_rank && time_keeper.check_time_limit(quiet);
    // let mut incumbent = None;

    for destination_rank in 0..n_ranks as Rank {
        if destination_rank != this_rank {
            let message = LocalLayerMessage::<T> {
                pruned,
                is_empty: layered_beams.is_empty(),
                time_out,
                bound: None,
                cost: None,
            };
            message.send(communicator, destination_rank, TAG_LOCAL_LAYER_MESSAGE);
        }
    }

    //let mut id_to_chain_node = vec![];
    //let mut node_communicator =
    //    NodeDepthCommunicator::<_, M, T>::new(communicator, TAG_NODE, model.clone(), capacity);
    let any_process = communicator.any_process();

    loop {}
}
