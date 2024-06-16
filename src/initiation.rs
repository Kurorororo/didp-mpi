use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    self,
    data_structure::{self, HashableSignatureVariables, StateInformation},
    util::{self, TimeKeeper},
    SearchInput, Solution, StateInRegistry, StateRegistry, TransitionWithId,
};
use std::collections::BinaryHeap;
use std::fmt::{Debug, Display};
use std::iter::Iterator;
use std::rc::Rc;
use std::str::FromStr;

use crate::bfs_node_with_distributed_id_chain::NodeGenerationResult;
use crate::distributed_id_chain::DistributedTransitionIdChain;
use crate::statistics::Statistics;
use crate::{
    bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain,
    distributed_id_chain::GetDistributedTransitionIdChain,
};

#[derive(Clone, Debug, Default)]
pub struct InitiationParameters {
    pub time_limit: Option<f64>,
    pub node_limit: Option<usize>,
    pub quiet: bool,
}

pub struct InitiationResult<T, N, V>
where
    T: Numeric,
    N: StateInformation<T>,
    V: TransitionInterface,
{
    pub n_nodes: usize,
    pub nodes: Vec<(Rc<HashableSignatureVariables>, Vec<Rc<N>>)>,
    pub id_to_chain_node: Vec<Rc<DistributedTransitionIdChain>>,
    pub id_to_depth: Vec<usize>,
    pub solution: Solution<T, TransitionWithId<V>>,
    pub statistics: Statistics,
}

fn construct_solution<V>(
    last: &TransitionWithId<V>,
    chain: &DistributedTransitionIdChain,
    transitions: &[TransitionWithId<V>],
    forced_transitions: &[TransitionWithId<V>],
    id_to_chain_node: &[Rc<DistributedTransitionIdChain>],
    result: &mut Vec<TransitionWithId<V>>,
) where
    V: TransitionInterface + Clone + Default,
{
    let (mut ids, mut forced, _) = chain.get_transition_ids_in_this_rank(id_to_chain_node);
    ids.reverse();
    forced.reverse();
    ids.push(last.id);
    forced.push(last.forced);

    for (id, forced) in ids.into_iter().zip(forced) {
        if forced {
            result.push(forced_transitions[id].clone());
        } else {
            result.push(transitions[id].clone());
        }
    }
}

fn extract_nodes<T, N>(
    registry: &mut StateRegistry<T, N>,
    primal_bound: Option<T>,
) -> Vec<(Rc<HashableSignatureVariables>, Vec<Rc<N>>)>
where
    T: Numeric + Ord + Display,
    <T as FromStr>::Err: Debug,
    N: BfsNodeWithDistributedIdChain<T>,
{
    let model = registry.model().clone();

    registry
        .drain()
        .flat_map(|(signature, nodes)| {
            let nodes = nodes
                .into_iter()
                .filter(|node| {
                    if let Some(dual_bound) = node.bound(&model) {
                        !data_structure::exceed_bound(&model, dual_bound, primal_bound)
                    } else {
                        true
                    }
                })
                .collect::<Vec<_>>();

            if nodes.is_empty() {
                None
            } else {
                Some((signature, nodes))
            }
        })
        .collect()
}

pub fn cbfs_initiator<T, N, E, B, V>(
    input: SearchInput<'_, N, TransitionWithId<V>>,
    mut successor_evaluator: E,
    mut base_cost_evaluator: B,
    parameters: InitiationParameters,
) -> InitiationResult<T, N, V>
where
    T: Numeric + Ord + Display,
    <T as FromStr>::Err: Debug,
    N: BfsNodeWithDistributedIdChain<T>,
    E: FnMut(
        StateInRegistry,
        T,
        &TransitionWithId<V>,
        &DistributedTransitionIdChain,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> NodeGenerationResult<Rc<N>>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
{
    let time_keeper = parameters
        .time_limit
        .map_or_else(TimeKeeper::default, |time_limit| {
            TimeKeeper::with_time_limit(time_limit)
        });
    let quiet = parameters.quiet;
    let node_limit = parameters.node_limit;

    let suffix = input.solution_suffix;
    let generator = input.generator;
    let model = &generator.model;

    let forced_transitions = generator
        .forced_transitions
        .iter()
        .map(|t| t.as_ref().clone())
        .collect::<Vec<_>>();
    let transitions = generator
        .transitions
        .iter()
        .map(|t| t.as_ref().clone())
        .collect::<Vec<_>>();

    let mut open = vec![BinaryHeap::with_capacity(1)];
    let mut registry = StateRegistry::new(model.clone());
    let mut id_to_chain_node = Vec::new();
    let mut id_to_depth = Vec::new();

    if let Some(node_limit) = node_limit {
        registry.reserve(node_limit);
        id_to_chain_node.reserve(node_limit);
        id_to_depth.reserve(node_limit);
    }

    let mut solution = Solution::default();
    let mut statistics = Statistics::default();
    let mut registry_size = 0;

    let node = input.node.and_then(|node| {
        let result = search_algorithm::rollout(
            node.state(),
            node.cost(model),
            suffix,
            &mut base_cost_evaluator,
            model,
        );
        result.and_then(|result| {
            if result.is_base {
                solution.cost = Some(result.cost);
                solution.time = time_keeper.elapsed_time();
                None
            } else {
                Some(node)
            }
        })
    });

    if let Some(node) = node {
        let result = registry.insert(node);
        let node = result.information.unwrap();
        open[0].push(node);
        solution.generated += 1;
        statistics.generated += 1;
        registry_size += 1;

        if !quiet {
            solution.time = time_keeper.elapsed_time();
            util::print_dual_bound(&solution);
        }
    } else {
        solution.is_infeasible = true;
    }

    let mut current_depth = 0;
    let mut no_node = true;
    let mut dual_bound_candidate = None;

    loop {
        if time_keeper.check_time_limit(quiet) {
            break;
        }

        if let Some(node_limit) = node_limit {
            if registry_size >= node_limit {
                break;
            }
        }

        if let Some(node) = open[current_depth].pop() {
            if node.is_closed() {
                continue;
            }

            node.close();

            if let Some(dual_bound) = node.bound(model) {
                if data_structure::exceed_bound(model, dual_bound, solution.cost) {
                    if N::ordered_by_bound() {
                        open[current_depth].clear();
                    }
                    continue;
                }
            }

            if no_node {
                no_node = false;
            }

            solution.expanded += 1;
            statistics.expanded += 1;

            node.get_distributed_transition_id_chain()
                .id
                .set(Some(id_to_chain_node.len()));
            id_to_chain_node.push(node.get_rc_distributed_transition_id_chain().clone());
            id_to_depth.push(current_depth);

            let mut no_successor = false;

            for transition in generator.applicable_transitions(node.state()) {
                if let Some((successor_state, g)) = model.generate_successor_state(
                    node.state(),
                    node.cost(model),
                    transition.as_ref(),
                    None,
                ) {
                    let result = search_algorithm::rollout(
                        &successor_state,
                        g,
                        suffix,
                        &mut base_cost_evaluator,
                        model,
                    );

                    if let Some(result) = result {
                        if result.is_base {
                            if !data_structure::exceed_bound(model, result.cost, solution.cost) {
                                construct_solution(
                                    transition.as_ref(),
                                    node.get_distributed_transition_id_chain(),
                                    &transitions,
                                    &forced_transitions,
                                    &id_to_chain_node,
                                    &mut solution.transitions,
                                );
                                solution.cost = Some(result.cost);
                                solution.time = time_keeper.elapsed_time();

                                if !quiet {
                                    util::print_primal_bound(&solution);
                                }
                            }

                            continue;
                        }
                    }

                    let result = successor_evaluator(
                        successor_state,
                        g,
                        &transition,
                        node.get_distributed_transition_id_chain(),
                        &mut registry,
                        solution.cost,
                    );
                    statistics.dominated_before_closed += result.dominated_before_closed;
                    statistics.dominated_after_closed += result.dominated_after_closed;

                    if let Some(successor) = result.node {
                        if result.dominated_before_closed == 0 && result.dominated_after_closed == 0
                        {
                            solution.generated += 1;
                            statistics.generated += 1;
                            registry_size += 1;
                        } else {
                            registry_size = registry_size + 1
                                - result.dominated_before_closed
                                - result.dominated_after_closed;
                        }

                        if no_successor {
                            no_successor = false;
                        }

                        while current_depth + 1 >= open.len() {
                            open.push(BinaryHeap::new());
                        }

                        open[current_depth + 1].push(successor);
                    }
                }
            }

            if no_successor {
                node.get_distributed_transition_id_chain().id.set(None);
                id_to_chain_node.pop();
                id_to_depth.pop();
            }
        }

        if N::ordered_by_bound() {
            if let Some(bound) = open[current_depth]
                .peek()
                .map(|node| node.bound(model).unwrap())
            {
                if !data_structure::exceed_bound(model, bound, dual_bound_candidate) {
                    dual_bound_candidate = Some(bound);
                }
            }
        }

        if no_node && current_depth + 1 == open.len() {
            solution.is_infeasible = solution.cost.is_none();
            solution.is_optimal = solution.cost.is_some();
            solution.best_bound = solution.cost;

            break;
        } else if current_depth + 1 == open.len() {
            if let Some(dual_bound) = dual_bound_candidate {
                if data_structure::exceed_bound(model, dual_bound, solution.cost) {
                    solution.is_optimal = solution.cost.is_some();
                    solution.best_bound = solution.cost;

                    break;
                } else {
                    solution.time = time_keeper.elapsed_time();
                    util::update_bound_if_better(&mut solution, dual_bound, model, quiet);
                    dual_bound_candidate = None;
                }
            }

            current_depth = 0;
            no_node = true;
        } else {
            current_depth += 1;
        }
    }

    if !solution.is_optimal && !solution.is_infeasible {
        solution.best_bound = dual_bound_candidate;
    }

    solution.time = time_keeper.elapsed_time();

    InitiationResult {
        n_nodes: registry_size,
        nodes: extract_nodes(&mut registry, solution.cost),
        id_to_chain_node,
        id_to_depth,
        solution,
        statistics,
    }
}

pub fn get_depth<N: GetDistributedTransitionIdChain>(node: &N, id_to_depth: &[usize]) -> usize {
    if let Some(id) = node.get_distributed_transition_id_chain().id.get() {
        id_to_depth[id]
    } else if let Some(id) = node.get_distributed_transition_id_chain().get_parent_id() {
        id_to_depth[id] + 1
    } else {
        0
    }
}
