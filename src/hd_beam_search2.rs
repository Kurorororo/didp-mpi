use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure::{exceed_bound, Beam};
use dypdl_heuristic_search::search_algorithm::util::TimeKeeper;
use dypdl_heuristic_search::search_algorithm::{
    get_solution_cost_and_suffix, BeamSearchParameters, BfsNode, SearchInput, Solution,
    StateInRegistry, StateRegistry,
};
use mpi::traits::*;
use mpi::{topology::SimpleCommunicator, Rank, Tag};
use std::error::Error;
use std::fmt::Display;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::{cmp, mem, thread};

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::is_float::IsFloat;
use crate::node_data_type::NodeDatatype;
use crate::partial_solution::PartialSolutionTags;
use crate::state_serializer::{self, StateSerializer};

struct LocalLayerMessageTags {
    tag_flags: Tag,
    tag_bound: Tag,
    tag_cost: Tag,
}

const TAG_NODE: Tag = 0;
const TAG_ALL_NODES_SENT: Tag = 1;
const TAG_PARTIAL_SOLUTION_REQUEST: Tag = 2;
const LOCAL_LAYER_MESSAGE_TAGS: LocalLayerMessageTags = LocalLayerMessageTags {
    tag_flags: 3,
    tag_bound: 4,
    tag_cost: 5,
};
const PARTIAL_SOLUTION_TAGS: PartialSolutionTags = PartialSolutionTags {
    tag_n: 6,
    tag_ids: 7,
    tag_forced: 8,
    tag_has_parent: 9,
    tag_parent_rank: 10,
};

struct NodeCommunicatorInner<'a, C, M, T> {
    model: &'a Model,
    communicator: &'a C,
    state_serializer: StateSerializer,
    tmp_buffer: Vec<u8>,
    _phantom: PhantomData<(M, T)>,
}

impl<'a, C, M, T> NodeCommunicatorInner<'a, C, M, T>
where
    C: Communicator,
    M: NodeDatatype<T>,
    T: Numeric + IsFloat,
{
    fn new(communicator: &'a C, model: &'a Model) -> Self {
        let state_serializer = StateSerializer::with_model(model);
        let tmp_buffer = vec![0; M::get_total_size(&state_serializer)];

        Self {
            model,
            communicator,
            state_serializer,
            tmp_buffer,
            _phantom: PhantomData,
        }
    }

    fn send(&mut self, destination_rank: Rank, node: M) {
        node.serialize_to(&self.state_serializer, &mut self.tmp_buffer);
        let destination = self.communicator.process_at_rank(destination_rank);
        destination.send_with_tag(&self.tmp_buffer, TAG_NODE);
    }

    fn receive(&mut self, source_rank: Rank, primal_bound: Option<T>) -> Option<M> {
        let source = self.communicator.process_at_rank(source_rank);
        source.receive_into_with_tag(&mut self.tmp_buffer, TAG_NODE);

        if let Some(bound) = M::get_bound(self.model, &self.state_serializer, &self.tmp_buffer) {
            if exceed_bound(self.model, bound, primal_bound) {
                return None;
            }
        }

        Some(M::deserialize(&self.state_serializer, &self.tmp_buffer))
    }
}

struct NodeCommunicator<'a, C, M, T> {
    communicator: NodeCommunicatorInner<'a, C, M, T>,
    buffers: Vec<Vec<M>>,
    channel_open: Vec<bool>,
}

impl<'a, C, M, T> NodeCommunicator<'a, C, M, T>
where
    C: Communicator,
    M: NodeDatatype<T>,
    T: IsFloat,
{
    fn new(communicator: &'a C, model: &'a Model, capacity: usize) -> Self {
        let inner = NodeCommunicatorInner::new(communicator, model);

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

struct LocalLayerMessage<T> {
    pruned: bool,
    is_empty: bool,
    time_out: bool,
    bound: Option<T>,
    cost: Option<T>,
}

impl<T: IsFloat> LocalLayerMessage<T> {
    fn send<C: Communicator>(
        &self,
        communicator: &C,
        destination_rank: Rank,
        tags: &LocalLayerMessageTags,
    ) {
        let destination = communicator.process_at_rank(destination_rank);
        let buffer = [
            self.pruned,
            self.is_empty,
            self.time_out,
            self.bound.is_some(),
            self.cost.is_some(),
        ];

        destination.send_with_tag(&buffer[..], tags.tag_flags);

        if T::is_float() {
            if let Some(bound) = self.bound {
                let bound = bound.to_continuous();
                destination.send_with_tag(&bound, tags.tag_bound);
            }

            if let Some(cost) = self.cost {
                let cost = cost.to_continuous();
                destination.send_with_tag(&cost, tags.tag_cost);
            }
        } else {
            if let Some(bound) = self.bound {
                let bound = bound.to_integer();
                destination.send_with_tag(&bound, tags.tag_bound);
            }

            if let Some(cost) = self.cost {
                let cost = cost.to_integer();
                destination.send_with_tag(&cost, tags.tag_cost);
            }
        }
    }

    fn receive<C: Communicator>(
        communicator: &C,
        source_rank: Rank,
        tags: &LocalLayerMessageTags,
    ) -> Self {
        let source = communicator.process_at_rank(source_rank);
        let mut buffer = [false; 5];
        source.receive_into_with_tag(&mut buffer[..], tags.tag_flags);

        let pruned = buffer[0];
        let is_empty = buffer[1];
        let time_out = buffer[2];
        let has_bound = buffer[3];
        let has_cost = buffer[4];

        let (bound, cost) = if T::is_float() {
            let bound = if has_bound {
                let mut bound = Continuous::default();
                source.receive_into_with_tag(&mut bound, tags.tag_bound);
                Some(T::from(bound))
            } else {
                None
            };
            let cost = if has_cost {
                let mut cost = Continuous::default();
                source.receive_into_with_tag(&mut cost, tags.tag_cost);
                Some(T::from(cost))
            } else {
                None
            };
            (bound, cost)
        } else {
            let bound = if has_bound {
                let mut bound = Integer::default();
                source.receive_into_with_tag(&mut bound, tags.tag_bound);
                Some(T::from(bound))
            } else {
                None
            };
            let cost = if has_cost {
                let mut cost = Integer::default();
                source.receive_into_with_tag(&mut cost, tags.tag_cost);
                Some(T::from(cost))
            } else {
                None
            };
            (bound, cost)
        };

        Self {
            pruned,
            is_empty,
            time_out,
            bound,
            cost,
        }
    }
}

pub fn hd_beam_search2<'a, T, N, M, E, B, V, F>(
    input: &'a SearchInput<'a, M, V>,
    transition_evaluator: E,
    base_cost_evaluator: B,
    parameters: BeamSearchParameters<T>,
    hash_function: F,
    communicator: &SimpleCommunicator,
) -> Solution<T, V>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T>,
    N: From<M>,
    M: Clone + NodeDatatype<T>,
    E: Fn(&N, Arc<V>, Option<T>) -> Option<M>,
    B: Fn(T, T) -> T,
    F: Fn(&M) -> u64,
    V: TransitionInterface + Clone + Default,
{
    let this_rank = communicator.rank();
    let time_keeper = if this_rank == 0 {
        Some(
            parameters
                .parameters
                .time_limit
                .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit),
        )
    } else {
        None
    };

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
    let mut generated = 0;

    if let Some(node) = input.node.clone() {
        let hash_value = hash_function(&node);
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
    // let mut best_dual_bound = None;
    let mut layer_dual_bound = None;
    let mut time_out = time_keeper
        .as_ref()
        .map_or(false, |time_keeper| time_keeper.check_time_limit(quiet));
    let mut incumbent: Option<(N, T, &[V])> = None;

    for destination_rank in 0..n_ranks as Rank {
        if destination_rank != this_rank {
            let message = LocalLayerMessage::<T> {
                pruned,
                is_empty: current_beam.is_empty(),
                time_out,
                bound: layer_dual_bound,
                cost: None,
            };
            message.send(communicator, destination_rank, &LOCAL_LAYER_MESSAGE_TAGS);
        }
    }

    let mut node_communicator = NodeCommunicator::<_, M, T>::new(communicator, model, capacity);

    loop {
        let mut is_empty = current_beam.is_empty();
        let mut cost = incumbent.as_ref().map(|(_, cost, _)| *cost);
        let mut goal_rank = if cost.is_some() {
            Some(this_rank)
        } else {
            None
        };

        {
            let previously_pruned = pruned;
            let mut previous_layer_dual_bound = layer_dual_bound;
            layer_dual_bound = None;

            let mut opened = 0;
            let mut sent_all = false;
            let mut expanded_all = false;
            let mut received_all = 0;
            let mut iter = current_beam.drain();

            while !sent_all || received_all < n_ranks - 1 {
                if received_all < n_ranks - 1 {
                    let any_process = communicator.any_process();

                    while let Some(status) = any_process.immediate_probe_with_tag(TAG_NODE) {
                        if let Some(node) =
                            node_communicator.receive(status.source_rank(), primal_bound)
                        {
                            let node = N::from(node);

                            let (new_generated, beam_pruning) =
                                next_beam.insert(&mut registry, node);

                            if !pruned && beam_pruning {
                                pruned = true;
                            }

                            if new_generated {
                                generated += 1;
                            }
                        }
                    }

                    while any_process
                        .immediate_probe_with_tag(TAG_ALL_NODES_SENT)
                        .is_some()
                    {
                        received_all += 1;
                    }
                }
            }
        }
    }

    Solution::default()
}
