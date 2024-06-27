use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::{self, StateInformation},
    StateRegistry,
};
use std::rc::Rc;
use std::{collections::BinaryHeap, fmt::Display};

pub struct LayeredBeams<T, N>
where
    T: Numeric,
    N: StateInformation<T>,
{
    model: Rc<Model>,
    beam_size: usize,
    current_minimum_depth: usize,
    depth_to_popped: Vec<usize>,
    depth_to_generated_all: Vec<bool>,
    depth_to_open: Vec<Option<BinaryHeap<Rc<N>>>>,
    depth_to_registry: Vec<Option<StateRegistry<T, N>>>,
    is_pruned: bool,
    best_discarded_bound: Option<T>,
}

pub struct PopResult<N> {
    pub node: N,
    pub depth: usize,
    pub is_last: bool,
}

impl<T, N> LayeredBeams<T, N>
where
    T: Numeric + Display,
    N: BfsNodeWithDistributedIdChain<T>,
{
    pub fn new(model: Rc<Model>, beam_size: usize) -> Self {
        Self {
            model,
            beam_size,
            current_minimum_depth: 0,
            depth_to_popped: vec![],
            depth_to_generated_all: vec![],
            depth_to_open: vec![],
            depth_to_registry: vec![],
            is_pruned: false,
            best_discarded_bound: None,
        }
    }

    pub fn current_minimum_depth(&self) -> usize {
        self.current_minimum_depth
    }

    pub fn best_discarded_bound(&self) -> Option<T> {
        self.best_discarded_bound
    }

    pub fn get_local_dual_bound(&self) -> Option<T> {
        None
    }

    pub fn update_best_discarded_bound(&mut self, bound: T) {
        if !data_structure::exceed_bound(&self.model, bound, self.best_discarded_bound) {
            self.best_discarded_bound = Some(bound);
        }
    }

    pub fn is_pruned(&self) -> bool {
        self.is_pruned
    }

    pub fn set_complete(&mut self, depth: usize) {
        self.depth_to_generated_all[depth] = true;
    }

    pub fn is_empty(&self) -> bool {
        for open in self.depth_to_open[self.current_minimum_depth..].iter() {
            if !open.as_ref().unwrap().is_empty() {
                return false;
            }
        }

        true
    }

    fn clean_garbage(open: &mut BinaryHeap<Rc<N>>) {
        while let Some(peak) = open.peek() {
            if peak.is_closed() {
                open.pop();
            } else {
                break;
            }
        }
    }

    pub fn get_registry_mut(&mut self, depth: usize) -> &mut StateRegistry<T, N> {
        while self.depth_to_open.len() < depth + 1 {
            self.depth_to_open.push(Some(BinaryHeap::new()));
            self.depth_to_registry
                .push(Some(StateRegistry::new(self.model.clone())));
            self.depth_to_generated_all.push(false);
            self.depth_to_popped.push(0);
        }

        self.depth_to_registry[depth].as_mut().unwrap()
    }

    pub fn insert(&mut self, node: Rc<N>, depth: usize) {
        debug_assert!(depth >= self.current_minimum_depth);

        self.depth_to_open[depth].as_mut().unwrap().push(node);
    }

    pub fn insert_with<F>(&mut self, node_generator: F, depth: usize)
    where
        F: FnOnce(&mut StateRegistry<T, N>) -> Option<Rc<N>>,
    {
        debug_assert!(depth >= self.current_minimum_depth);

        let registry = self.get_registry_mut(depth);

        if let Some(node) = node_generator(registry) {
            self.insert(node, depth);
        }
    }

    fn pop_from_open(
        open: &mut BinaryHeap<Rc<N>>,
        model: &Model,
        primal_bound: Option<T>,
    ) -> Option<Rc<N>> {
        while let Some(node) = open.pop() {
            if node.bound(model).map_or(false, |bound| {
                data_structure::exceed_bound(model, bound, primal_bound)
            }) {
                if N::ordered_by_bound() {
                    open.clear();

                    return None;
                }

                Self::clean_garbage(open);
                continue;
            }

            Self::clean_garbage(open);
            return Some(node);
        }

        None
    }

    pub fn pop(&mut self, primal_bound: Option<T>) -> Option<PopResult<Rc<N>>> {
        let mut depth = self.current_minimum_depth;

        while depth < self.depth_to_open.len() {
            let (node, is_empty) = {
                let open = self.depth_to_open[depth].as_mut().unwrap();

                (
                    Self::pop_from_open(open, &self.model, primal_bound),
                    open.is_empty(),
                )
            };

            if let Some(node) = node {
                self.depth_to_popped[depth] += 1;
                let is_last = self.depth_to_popped[depth] == self.beam_size
                    || (is_empty && self.depth_to_generated_all[depth]);

                if is_last {
                    for d in self.current_minimum_depth..=depth {
                        let open = self.depth_to_open[d].take().unwrap();
                        self.depth_to_registry[d].take();

                        if N::ordered_by_bound() {
                            if let Some(bound) =
                                open.peek().and_then(|peek| peek.bound(&self.model))
                            {
                                self.update_best_discarded_bound(bound);
                            }
                        }

                        if !self.is_pruned && (!self.depth_to_generated_all[d] || !open.is_empty())
                        {
                            self.is_pruned = true;
                        }
                    }

                    self.current_minimum_depth = depth + 1;
                }

                return Some(PopResult {
                    node,
                    depth,
                    is_last,
                });
            }

            depth += 1;
        }

        None
    }
}

pub struct LayeredPerRankCounters {
    n_ranks: usize,
    counters: Vec<Option<Vec<i32>>>,
    minimum_depth: usize,
}

impl LayeredPerRankCounters {
    pub fn new(n_ranks: usize) -> Self {
        Self {
            n_ranks,
            counters: Vec::new(),
            minimum_depth: 0,
        }
    }

    pub fn get_counter(&mut self, depth: usize) -> &mut [i32] {
        while self.counters.len() <= depth {
            self.counters.push(Some(vec![]));
        }

        let counters = self.counters[depth].as_mut().unwrap();

        if counters.is_empty() {
            counters.resize(self.n_ranks, 0);
        }

        &mut counters[..]
    }

    pub fn increment(&mut self, rank: usize, depth: usize) -> bool {
        let counters = self.get_counter(depth);
        counters[rank] += 1;

        counters[rank] == 0
    }

    pub fn decrease(&mut self, rank: usize, depth: usize, n: i32) -> bool {
        let counters = self.get_counter(depth);
        counters[rank] -= n;

        counters[rank] == 0
    }

    pub fn get_count(&self, rank: usize, depth: usize) -> i32 {
        self.counters[depth].as_ref().unwrap()[rank]
    }

    pub fn get_minimum_depth(&self) -> usize {
        self.minimum_depth
    }

    pub fn clear_to_depth(&mut self, depth: usize) {
        for counters in &mut self.counters[self.minimum_depth..=depth] {
            *counters = None;
        }

        self.minimum_depth = depth + 1;
    }
}
