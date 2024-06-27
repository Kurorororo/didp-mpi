use crate::bfs_node_with_distributed_id_chain::{
    BfsNodeWithDistributedIdChain, NodeGenerationResult,
};
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
    generated: usize,
    dominated_before_expanded: usize,
    dominated_after_expanded: usize,
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
        let depth_to_popped = vec![0];
        let depth_to_complete = vec![false];
        let mut depth_to_open = Vec::with_capacity(1);
        let mut depth_to_registry = Vec::with_capacity(1);

        depth_to_open.push(Some(BinaryHeap::new()));
        depth_to_registry.push(Some(StateRegistry::new(model.clone())));

        Self {
            model,
            beam_size,
            current_minimum_depth: 0,
            depth_to_popped,
            depth_to_generated_all: depth_to_complete,
            depth_to_open,
            depth_to_registry,
            is_pruned: false,
            generated: 0,
            dominated_before_expanded: 0,
            dominated_after_expanded: 0,
        }
    }

    pub fn current_minimum_depth(&self) -> usize {
        self.current_minimum_depth
    }

    pub fn generated(&self) -> usize {
        self.generated
    }

    pub fn dominated_before_expanded(&self) -> usize {
        self.dominated_before_expanded
    }

    pub fn dominated_after_expanded(&self) -> usize {
        self.dominated_after_expanded
    }

    pub fn set_complete(&mut self, depth: usize) {
        self.depth_to_generated_all[depth] = true;
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

    pub fn insert_with<F>(&mut self, node_generator: F, depth: usize)
    where
        F: FnOnce(&mut StateRegistry<T, N>) -> NodeGenerationResult<Rc<N>>,
    {
        debug_assert!(depth >= self.current_minimum_depth);

        while self.depth_to_open.len() < depth + 1 {
            self.depth_to_open.push(Some(BinaryHeap::new()));
            self.depth_to_registry
                .push(Some(StateRegistry::new(self.model.clone())));
            self.depth_to_generated_all.push(false);
            self.depth_to_popped.push(0);
        }

        let registry = self.depth_to_registry[depth].as_mut().unwrap();
        let result = node_generator(registry);

        if result.dominated_before_closed > 0 {
            self.dominated_before_expanded += result.dominated_before_closed;
            Self::clean_garbage(self.depth_to_open[depth].as_mut().unwrap());
        }

        self.dominated_after_expanded += result.dominated_after_closed;

        if let Some(node) = result.node {
            if result.dominated_before_closed == 0 && result.dominated_after_closed == 0 {
                self.generated += 1;
            }

            self.depth_to_open[depth].as_mut().unwrap().push(node);
        }
    }

    pub fn insert(&mut self, node: N, depth: usize) {
        let node_generator = |registry: &mut StateRegistry<T, N>| {
            let result = registry.insert(node);
            let mut dominated_before_expanded = 0;
            let mut dominated_after_expanded = 0;

            for dominated in result.dominated.iter() {
                if dominated.is_closed() {
                    dominated_after_expanded += 1;
                } else {
                    dominated.close();
                    dominated_before_expanded += 1;
                }
            }

            NodeGenerationResult {
                node: result.information,
                is_pruned_by_bound: false,
                dominated_before_closed: dominated_before_expanded,
                dominated_after_closed: dominated_after_expanded,
            }
        };

        self.insert_with(node_generator, depth);
    }

    fn pop_from_open(
        open: &mut BinaryHeap<Rc<N>>,
        model: &Model,
        primal_bound: Option<T>,
    ) -> Option<Rc<N>> {
        while let Some(node) = open.pop() {
            if let Some(bound) = node.bound(model) {
                if data_structure::exceed_bound(model, bound, primal_bound) {
                    if N::ordered_by_bound() {
                        open.clear();

                        return None;
                    }

                    Self::clean_garbage(open);
                    continue;
                }
            }

            Self::clean_garbage(open);

            return Some(node);
        }

        None
    }

    pub fn pop(&mut self, primal_bound: Option<T>) -> Option<PopResult<Rc<N>>> {
        let mut depth = self.current_minimum_depth;

        while depth < self.depth_to_open.len() {
            let open = self.depth_to_open[depth].as_mut().unwrap();

            if let Some(node) = Self::pop_from_open(open, &self.model, primal_bound) {
                self.depth_to_popped[depth] += 1;
                let is_last = self.depth_to_popped[depth] == self.beam_size
                    || open.is_empty() && self.depth_to_generated_all[depth];

                if is_last {
                    for d in self.current_minimum_depth..=depth {
                        self.depth_to_open[d] = None;
                        self.depth_to_registry[d] = None;

                        if !self.is_pruned
                            && (!self.depth_to_generated_all[d]
                                || !self.depth_to_open[d].as_ref().unwrap().is_empty())
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

    pub fn is_empty(&self) -> bool {
        for open in self.depth_to_open[self.current_minimum_depth..].iter() {
            if !open.as_ref().unwrap().is_empty() {
                return false;
            }
        }

        true
    }
}
