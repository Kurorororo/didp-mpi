use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::{
    search_algorithm::{
        data_structure::{self, BfsNode},
        get_solution_cost_and_suffix,
        util::{self, TimeKeeper},
        SearchInput, StateRegistry, SuccessorGenerator,
    },
    Parameters, Search, Solution,
};
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt;
use std::hash::Hash;
use std::rc::Rc;

pub struct Hcbfs<'a, T, N, E, B, V = Transition>
where
    T: Numeric + Ord + fmt::Display + Hash,
    N: BfsNode<T, V>,
    E: FnMut(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    generator: SuccessorGenerator<V>,
    suffix: &'a [V],
    transition_evaluator: E,
    base_cost_evaluator: B,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    open: BinaryHeap<(Rc<N>, usize)>,
    layered_open: Vec<BinaryHeap<Rc<N>>>,
    registry: StateRegistry<T, N>,
    is_layered_turn: bool,
    current_depth: usize,
    time_keeper: TimeKeeper,
    count_bound_to_expanded: bool,
    bound_to_expanded: FxHashMap<T, usize>,
    solution: Solution<T>,
}

impl<'a, T, N, E, B, V> Hcbfs<'a, T, N, E, B, V>
where
    T: Numeric + Ord + fmt::Display + Hash,
    N: BfsNode<T, V>,
    E: FnMut(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    /// Create a new HCBFS solver.
    pub fn new(
        input: SearchInput<'a, N, V>,
        transition_evaluator: E,
        base_cost_evaluator: B,
        parameters: Parameters<T>,
        count_bound_to_expanded: bool,
    ) -> Hcbfs<'a, T, N, E, B, V> {
        let time_keeper = parameters
            .time_limit
            .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
        let primal_bound = parameters.primal_bound;
        let get_all_solutions = parameters.get_all_solutions;
        let quiet = parameters.quiet;

        let layered_open = vec![BinaryHeap::new()];
        let mut open = BinaryHeap::new();
        let mut registry = StateRegistry::<_, _>::new(input.generator.model.clone());

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();

        if let Some(node) = input.node {
            let result = registry.insert(node);
            let node = result.information.unwrap();
            solution.generated += 1;
            solution.best_bound = node.bound(&input.generator.model);
            open.push((node, 0));

            if !quiet {
                solution.time = time_keeper.elapsed_time();
                util::print_dual_bound(&solution);
            }
        } else {
            solution.is_infeasible = true;
        }

        Hcbfs {
            generator: input.generator,
            suffix: input.solution_suffix,
            transition_evaluator,
            base_cost_evaluator,
            primal_bound,
            get_all_solutions,
            quiet,
            open,
            layered_open,
            registry,
            is_layered_turn: false,
            current_depth: 0,
            time_keeper,
            count_bound_to_expanded,
            bound_to_expanded: FxHashMap::default(),
            solution,
        }
    }

    fn pop_from_open(&mut self) -> Option<(Rc<N>, usize)> {
        let model = &self.generator.model;

        while let Some((node, depth)) = self.open.pop() {
            if node.is_closed() {
                continue;
            }

            node.close();

            if node.bound(model).map_or(false, |dual_bound| {
                data_structure::exceed_bound(model, dual_bound, self.primal_bound)
            }) {
                if N::ordered_by_bound() {
                    self.open.clear();
                }
            } else {
                if let Some(dual_bound) = node.bound(model) {
                    self.solution.time = self.time_keeper.elapsed_time();
                    util::update_bound_if_better(&mut self.solution, dual_bound, model, self.quiet);
                }

                return Some((node, depth));
            }
        }

        None
    }

    fn pop_from_layered_open(&mut self) -> Option<(Rc<N>, usize)> {
        let model = &self.generator.model;

        while let Some(node) = self.layered_open[self.current_depth].pop() {
            if node.is_closed() {
                continue;
            }

            node.close();

            if node.bound(model).map_or(false, |dual_bound| {
                data_structure::exceed_bound(model, dual_bound, self.primal_bound)
            }) {
                if N::ordered_by_bound() {
                    self.layered_open[self.current_depth].clear();
                }
            } else {
                return Some((node, self.current_depth));
            }
        }

        None
    }

    fn pop_node_and_depth(&mut self) -> Option<(Rc<N>, usize)> {
        if self.is_layered_turn {
            if self.current_depth > self.layered_open.len() - 1 {
                self.current_depth = 0;
            }

            let initial_depth = self.current_depth;

            loop {
                let result = self.pop_from_layered_open();
                self.current_depth += 1;

                if result.is_some() {
                    self.is_layered_turn = false;

                    return result;
                } else {
                    if self.current_depth > self.layered_open.len() - 1 {
                        self.current_depth = 0;
                    }

                    if self.current_depth == initial_depth {
                        break;
                    }
                }
            }
        }

        self.is_layered_turn = true;

        self.pop_from_open()
    }

    pub fn get_bound_to_expanded(&self) -> FxHashMap<T, usize> {
        self.bound_to_expanded.clone()
    }
}

impl<'a, T, N, E, B, V> Search<T> for Hcbfs<'a, T, N, E, B, V>
where
    T: Numeric + Ord + fmt::Display + Hash,
    N: BfsNode<T, V>,
    E: FnMut(&N, Rc<V>, &mut StateRegistry<T, N>, Option<T>) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        if self.solution.is_terminated() {
            return Ok((self.solution.clone(), true));
        }

        self.time_keeper.start();
        let suffix = self.suffix;

        while let Some((node, depth)) = self.pop_node_and_depth() {
            let model = &self.generator.model;

            if let Some((cost, suffix)) =
                get_solution_cost_and_suffix(model, &*node, suffix, &mut self.base_cost_evaluator)
            {
                if !data_structure::exceed_bound(model, cost, self.primal_bound) {
                    self.primal_bound = Some(cost);
                    let time = self.time_keeper.elapsed_time();
                    util::update_solution(
                        &mut self.solution,
                        &*node,
                        cost,
                        suffix,
                        time,
                        self.quiet,
                    );
                    self.time_keeper.stop();

                    return Ok((self.solution.clone(), self.solution.is_optimal));
                } else if self.get_all_solutions {
                    let mut solution = self.solution.clone();
                    let time = self.time_keeper.elapsed_time();
                    util::update_solution(&mut solution, &*node, cost, suffix, time, true);
                    self.time_keeper.stop();

                    return Ok((solution, false));
                }
                continue;
            }

            if self.time_keeper.check_time_limit(self.quiet) {
                self.solution.time_out = true;
                self.solution.time = self.time_keeper.elapsed_time();
                self.time_keeper.stop();

                return Ok((self.solution.clone(), true));
            }

            self.solution.expanded += 1;

            if self.count_bound_to_expanded {
                if let Some(bound) = node.bound(model) {
                    self.bound_to_expanded
                        .entry(bound)
                        .and_modify(|e| *e += 1)
                        .or_insert(1);
                }
            }

            for transition in self.generator.applicable_transitions(node.state()) {
                if let Some((successor, new_generated)) = (self.transition_evaluator)(
                    &node,
                    transition,
                    &mut self.registry,
                    self.primal_bound,
                ) {
                    while depth + 1 >= self.layered_open.len() {
                        self.layered_open.push(BinaryHeap::new());
                    }

                    self.open.push((successor.clone(), depth + 1));
                    self.layered_open[depth + 1].push(successor);

                    if new_generated {
                        self.solution.generated += 1;
                    }
                }
            }
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.best_bound = self.solution.cost;
        self.solution.time = self.time_keeper.elapsed_time();
        self.time_keeper.stop();
        Ok((self.solution.clone(), true))
    }
}
