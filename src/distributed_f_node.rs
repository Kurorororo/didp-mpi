use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::distributed_id_chain::GetDistributedTransitionIdChain;

use super::distributed_id_chain::DistributedTransitionIdChain;
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::{self, StateInformation},
    StateInRegistry, StateRegistry, TransitionWithId,
};
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::Display;
use std::rc::Rc;

/// Node ordered by the f-value and associated with a transition ids chain.
#[derive(Debug, Clone)]
pub struct DistributedFNode<T>
where
    T: Numeric,
{
    pub g: T,
    pub h: T,
    pub f: T,
    state: StateInRegistry,
    transition_id_chain: Rc<DistributedTransitionIdChain>,
    closed: Cell<bool>,
}

impl<T> DistributedFNode<T>
where
    T: Numeric,
{
    pub fn new(
        state: StateInRegistry,
        g: T,
        h: T,
        f: T,
        transition_id_chain: Rc<DistributedTransitionIdChain>,
    ) -> Self {
        Self {
            state,
            g,
            h,
            f,
            closed: Cell::new(false),
            transition_id_chain,
        }
    }

    /// Generates a root node.
    pub fn generate_root_node<S, H, F>(
        state: S,
        cost: T,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        StateInRegistry: From<S>,
        H: FnOnce(&StateInRegistry) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry) -> T,
    {
        let state = StateInRegistry::from(state);
        let h = h_evaluator(&state)?;
        let f = f_evaluator(cost, h, &state);

        if data_structure::exceed_bound(model, f, primal_bound) {
            return None;
        }

        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };

        Some(Self::new(
            state,
            cost,
            h,
            f,
            Rc::new(DistributedTransitionIdChain::default()),
        ))
    }

    /// Generates a successor node.
    pub fn generate_successor_node<V, H, F>(
        &self,
        transition: &TransitionWithId<V>,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        V: TransitionInterface,
        H: FnOnce(&StateInRegistry) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry) -> T,
    {
        let (state, g) = model.generate_successor_state(&self.state, self.g, transition, None)?;
        let h = h_evaluator(&state)?;
        let f = f_evaluator(g, h, &state);

        if data_structure::exceed_bound(model, f, primal_bound) {
            return None;
        }

        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };

        let transition_id_chain = Rc::new(
            self.transition_id_chain
                .generate_successor(transition.id, transition.forced),
        );

        Some(Self::new(state, g, h, f, transition_id_chain))
    }

    /// Inserts a successor node into the registry.
    pub fn insert_successor_node<V, H, F>(
        &self,
        transition: &TransitionWithId<V>,
        registry: &mut StateRegistry<T, Self>,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<(Rc<Self>, bool)>
    where
        V: TransitionInterface,
        H: FnOnce(&StateInRegistry) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry) -> T,
    {
        let (state, g) =
            registry
                .model()
                .generate_successor_state(&self.state, self.g, transition, None)?;

        let model = registry.model().clone();
        let maximize = model.reduce_function == ReduceFunction::Max;

        let constructor = |state, g, other: Option<&Self>| {
            let h = if let Some(other) = other {
                if maximize {
                    other.h
                } else {
                    -other.h
                }
            } else {
                h_evaluator(&state)?
            };
            let f = f_evaluator(g, h, &state);

            if data_structure::exceed_bound(&model, f, primal_bound) {
                return None;
            }

            let (h, f) = if maximize { (h, f) } else { (-h, -f) };

            let transition_id_chain = Rc::new(
                self.transition_id_chain
                    .generate_successor(transition.id, transition.forced),
            );

            Some(Self::new(state, g, h, f, transition_id_chain))
        };

        let result = registry.insert_with(state, g, constructor);

        for d in result.dominated.iter() {
            if !d.is_closed() {
                d.close();
            }
        }

        let successor = result.information?;

        Some((successor, result.dominated.is_empty()))
    }
}

impl<T> PartialEq for DistributedFNode<T>
where
    T: Numeric,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.h == other.h
    }
}

impl<T> Eq for DistributedFNode<T> where T: Numeric {}

impl<T> Ord for DistributedFNode<T>
where
    T: Numeric + Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => self.h.cmp(&other.h),
            result => result,
        }
    }
}

impl<T> PartialOrd for DistributedFNode<T>
where
    T: Numeric + Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> StateInformation<T> for DistributedFNode<T>
where
    T: Numeric,
{
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn state_mut(&mut self) -> &mut StateInRegistry {
        &mut self.state
    }

    #[inline]
    fn cost(&self, _: &Model) -> T {
        self.g
    }

    #[inline]
    fn bound(&self, model: &Model) -> Option<T> {
        if model.reduce_function == ReduceFunction::Min {
            Some(-self.f)
        } else {
            Some(self.f)
        }
    }

    #[inline]
    fn is_closed(&self) -> bool {
        self.closed.get()
    }

    #[inline]
    fn close(&self) {
        self.closed.set(true);
    }
}

impl<T> GetDistributedTransitionIdChain for DistributedFNode<T>
where
    T: Numeric,
{
    #[inline]
    fn get_distributed_transition_id_chain(&self) -> &Rc<DistributedTransitionIdChain> {
        &self.transition_id_chain
    }
}

impl<T> BfsNodeWithDistributedIdChain<T> for DistributedFNode<T>
where
    T: Numeric + Display + Ord,
{
    #[inline]
    fn ordered_by_bound() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_root_node_some_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_some());
        let mut node = node.unwrap();
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), Some(1));
        assert!(!node.is_closed());
        assert_eq!(
            node.transition_id_chain,
            Rc::new(DistributedTransitionIdChain::default())
        );
    }

    #[test]
    fn generate_root_node_some_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let mut expected_state = StateInRegistry::from(state.clone());
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_some());
        let mut node = node.unwrap();
        assert_eq!(node.state(), &expected_state);
        assert_eq!(node.state_mut(), &mut expected_state);
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), Some(1));
        assert!(!node.is_closed());
        assert_eq!(
            node.transition_id_chain,
            Rc::new(DistributedTransitionIdChain::default())
        );
    }

    #[test]
    fn generate_root_node_pruned_by_bound_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = Some(-1);

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_none());
    }

    #[test]
    fn generate_root_node_pruned_by_bound_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = Some(1);

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_none());
    }

    #[test]
    fn generate_root_node_pruned_by_h() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| None;
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_none());
    }

    #[test]
    fn close() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());

        let state = model.target.clone();
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_some());
        let node = node.unwrap();
        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn generate_successor_some_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };

        let state = model.target.clone();
        let mut expected_state = transition.apply(&state, &model.table_registry);
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let successor = node.generate_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
    }

    #[test]
    fn generate_successor_some_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };

        let state = model.target.clone();
        let mut expected_state = transition.apply(&state, &model.table_registry);
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let successor = node.generate_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_some());
        let mut successor = successor.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.state_mut(), &mut expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
    }

    #[test]
    fn generate_successor_pruned_by_constraint() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let result =
            model.add_state_constraint(Condition::comparison_i(ComparisonOperator::Le, v1, 0));
        assert!(result.is_ok());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };

        let state = model.target.clone();
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let successor = node.generate_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_none());
    }

    #[test]
    fn generate_successor_pruned_by_bound_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };

        let state = model.target.clone();
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = Some(2);

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));

        let successor = node.generate_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_none());
    }

    #[test]
    fn generate_successor_pruned_by_bound_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };

        let state = model.target.clone();
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(3);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = Some(3);

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));

        let h_evaluator = |_: &StateInRegistry| Some(1);
        let successor = node.generate_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_none());
    }

    #[test]
    fn generate_successor_pruned_by_h() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };

        let state = model.target.clone();
        let cost = 1;
        let h_evaluator = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateInRegistry| g + h;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));

        let h_evaluator = |_: &StateInRegistry| None;
        let successor = node.generate_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_none());
    }

    #[test]
    fn insert_successor_non_dominance_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };

        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, _: i32, _: &_| g;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            1,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let result = node.insert_successor_node(
            &transition,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert!(generated);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
    }

    #[test]
    fn insert_successor_non_dominance_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };

        let expected_state: StateInRegistry = transition.apply(&state, &model.table_registry);
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, _: i32, _: &_| g;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            1,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let result = node.insert_successor_node(
            &transition,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert!(generated);
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
    }

    #[test]
    fn insert_successor_pruned_by_constraint() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let result =
            model.add_state_constraint(Condition::comparison_i(ComparisonOperator::Le, v1, 0));
        assert!(result.is_ok());
        let model = Rc::new(model);

        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());

        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };
        let state = model.target.clone();

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, _: i32, _: &_| g;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            1,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let result = node.insert_successor_node(
            &transition,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_none());
    }

    #[test]
    fn insert_successor_dominating_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", false, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition1 = Transition::default();
        let result = transition1.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition1.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition1.set_cost(IntegerExpression::Cost + 1);
        let transition1 = TransitionWithId {
            id: 0,
            transition: transition1,
            forced: false,
        };

        let mut transition2 = Transition::default();
        let result = transition2.add_effect(v1, v1 + 2);
        assert!(result.is_ok());
        let result = transition2.add_effect(v2, v2 + 2);
        assert!(result.is_ok());
        transition2.set_cost(IntegerExpression::Cost + 1);
        let transition2 = TransitionWithId {
            id: 1,
            transition: transition2,
            forced: false,
        };

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, _: i32, _: &_| g;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            1,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let expected_state: StateInRegistry = transition1.apply(&state, &model.table_registry);
        let result = node.insert_successor_node(
            &transition1,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
        assert!(generated);

        let expected_state: StateInRegistry = transition2.apply(&state, &model.table_registry);
        let result = node.insert_successor_node(
            &transition2,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
        assert!(!generated);
    }

    #[test]
    fn inset_successor_dominating_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", false, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition1 = Transition::default();
        let result = transition1.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition1.add_effect(v2, v2 - 1);
        assert!(result.is_ok());
        transition1.set_cost(IntegerExpression::Cost + 1);
        let transition1 = TransitionWithId {
            id: 0,
            transition: transition1,
            forced: false,
        };

        let mut transition2 = Transition::default();
        let result = transition2.add_effect(v1, v1 + 2);
        assert!(result.is_ok());
        let result = transition2.add_effect(v2, v2 - 1);
        assert!(result.is_ok());
        transition2.set_cost(IntegerExpression::Cost + 1);
        let transition2 = TransitionWithId {
            id: 1,
            transition: transition2,
            forced: false,
        };

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, _: i32, _: &_| g;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            1,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let expected_state: StateInRegistry = transition1.apply(&state, &model.table_registry);
        let result = node.insert_successor_node(
            &transition1,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
        assert!(generated);

        let expected_state: StateInRegistry = transition2.apply(&state, &model.table_registry);
        let result = node.insert_successor_node(
            &transition2,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
        assert!(!generated);
    }

    #[test]
    fn insert_successor_dominated_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", false, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition1 = Transition::default();
        let result = transition1.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition1.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition1.set_cost(IntegerExpression::Cost + 1);
        let transition1 = TransitionWithId {
            id: 0,
            transition: transition1,
            forced: false,
        };

        let mut transition2 = Transition::default();
        let result = transition2.add_effect(v1, v1 + 2);
        assert!(result.is_ok());
        let result = transition2.add_effect(v2, v2 + 2);
        assert!(result.is_ok());
        transition2.set_cost(IntegerExpression::Cost + 1);
        let transition2 = TransitionWithId {
            id: 1,
            transition: transition2,
            forced: false,
        };

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, _: i32, _: &_| g;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            2,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let expected_state: StateInRegistry = transition2.apply(&state, &model.table_registry);
        let result = node.insert_successor_node(
            &transition2,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 3);
        assert_eq!(successor.bound(&model), Some(3));
        assert!(!successor.is_closed());
        assert!(generated);

        let result = node.insert_successor_node(
            &transition1,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_none());
    }

    #[test]
    fn insert_successor_dominated_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", false, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition1 = Transition::default();
        let result = transition1.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition1.add_effect(v2, v2 - 1);
        assert!(result.is_ok());
        transition1.set_cost(IntegerExpression::Cost + 1);
        let transition1 = TransitionWithId {
            id: 0,
            transition: transition1,
            forced: false,
        };

        let mut transition2 = Transition::default();
        let result = transition2.add_effect(v1, v1 + 2);
        assert!(result.is_ok());
        let result = transition2.add_effect(v2, v2 - 1);
        assert!(result.is_ok());
        transition2.set_cost(IntegerExpression::Cost + 1);
        let transition2 = TransitionWithId {
            id: 1,
            transition: transition2,
            forced: false,
        };

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, _: i32, _: &_| g;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            2,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let expected_state: StateInRegistry = transition2.apply(&state, &model.table_registry);
        let result = node.insert_successor_node(
            &transition2,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 3);
        assert_eq!(successor.bound(&model), Some(3));
        assert!(!successor.is_closed());
        assert!(generated);

        let result = node.insert_successor_node(
            &transition1,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_none());
    }

    #[test]
    fn insert_successor_pruned_by_bound_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", false, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition1 = Transition::default();
        let result = transition1.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition1.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition1.set_cost(IntegerExpression::Cost + 1);
        let transition1 = TransitionWithId {
            id: 0,
            transition: transition1,
            forced: false,
        };

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
        let primal_bound = Some(2);

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            1,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let result = node.insert_successor_node(
            &transition1,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_none());
    }

    #[test]
    fn insert_successor_pruned_by_bound_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", false, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = StateInRegistry::from(model.target.clone());
        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition1 = Transition::default();
        let result = transition1.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition1.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        transition1.set_cost(IntegerExpression::Cost + 1);
        let transition1 = TransitionWithId {
            id: 0,
            transition: transition1,
            forced: false,
        };

        let h_evaluator = |_: &_| Some(3);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
        let primal_bound = Some(3);

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            1,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));
        let result = registry.insert(node.clone());
        assert!(result.information.is_some());

        let h_evaluator = |_: &_| Some(1);
        let result = node.insert_successor_node(
            &transition1,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_none());
    }

    #[test]
    fn insert_successor_pruned_by_h() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", false, 1);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 1);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let result =
            model.add_state_constraint(Condition::comparison_i(ComparisonOperator::Le, v1, 0));
        assert!(result.is_ok());
        let model = Rc::new(model);

        let mut registry = StateRegistry::<_, _>::new(model.clone());

        let mut transition1 = Transition::default();
        let result = transition1.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition1.add_effect(v2, v2 + 1);
        assert!(result.is_ok());

        transition1.set_cost(IntegerExpression::Cost + 1);
        let transition1 = TransitionWithId {
            id: 0,
            transition: transition1,
            forced: false,
        };
        let state = model.target.clone();

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
        let primal_bound = None;

        let node = DistributedFNode::generate_root_node(
            state.clone(),
            1,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        node.transition_id_chain.id.set(Some(0));

        let h_evaluator = |_: &_| None;
        let result = node.insert_successor_node(
            &transition1,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(result.is_none());
    }

    #[test]
    fn cmp_min() {
        let mut model = Model::default();
        model.set_minimize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = model.target.clone();
        let h_evaluator_0 = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node1 = DistributedFNode::<_>::generate_root_node(
            state.clone(),
            0,
            &model,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node1.is_some());
        let node1 = Rc::new(node1.unwrap());
        node1.transition_id_chain.id.set(Some(0));

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };
        let node2 =
            node1.generate_successor_node(&transition, &model, &h_evaluator_0, &f_evaluator, None);
        assert!(node2.is_some());
        let node2 = Rc::new(node2.unwrap());

        let mut transition = Transition::default();
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 1,
            transition,
            forced: false,
        };
        let mut registry = StateRegistry::<_, DistributedFNode<_>>::new(model.clone());
        let node3 = node1.insert_successor_node(
            &transition,
            &mut registry,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node3.is_some());
        let (node3, _) = node3.unwrap();

        let h_evaluator_1 = |_: &StateInRegistry| Some(1);
        let node4 = DistributedFNode::<_>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator_1,
            &f_evaluator,
            None,
        );
        assert!(node4.is_some());
        let node4 = Rc::new(node4.unwrap());

        assert!(node1 == node1);
        assert!(node1 >= node1);
        assert!(node1 == node2);
        assert!(node1 >= node2);
        assert!(node1 <= node2);
        assert!(node1 >= node3);
        assert!(node1 > node3);
        assert!(node1 != node3);
        assert!(node1 >= node4);
        assert!(node1 > node4);
        assert!(node1 != node4);
        assert!(node3 >= node4);
        assert!(node3 > node4);
        assert!(node3 != node4);
    }

    #[test]
    fn cmp_max() {
        let mut model = Model::default();
        model.set_maximize();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let state = model.target.clone();
        let h_evaluator_0 = |_: &StateInRegistry| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node1 = DistributedFNode::<_>::generate_root_node(
            state.clone(),
            0,
            &model,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node1.is_some());
        let node1 = Rc::new(node1.unwrap());
        node1.transition_id_chain.id.set(Some(0));

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, v1 + 1);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 1);
        assert!(result.is_ok());
        let transition = TransitionWithId {
            id: 0,
            transition,
            forced: false,
        };
        let node2 =
            node1.generate_successor_node(&transition, &model, &h_evaluator_0, &f_evaluator, None);
        assert!(node2.is_some());
        let node2 = Rc::new(node2.unwrap());

        let mut transition = Transition::default();
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            id: 1,
            transition,
            forced: false,
        };
        let mut registry = StateRegistry::<_, DistributedFNode<_>>::new(model.clone());
        let node3 = node1.insert_successor_node(
            &transition,
            &mut registry,
            &h_evaluator_0,
            &f_evaluator,
            None,
        );
        assert!(node3.is_some());
        let (node3, _) = node3.unwrap();

        let h_evaluator_1 = |_: &StateInRegistry| Some(1);
        let node4 = DistributedFNode::<_>::generate_root_node(
            state,
            0,
            &model,
            &h_evaluator_1,
            &f_evaluator,
            None,
        );
        assert!(node4.is_some());
        let node4 = Rc::new(node4.unwrap());

        assert!(node1 == node1);
        assert!(node1 >= node1);
        assert!(node1 == node2);
        assert!(node1 >= node2);
        assert!(node1 <= node2);
        assert!(node1 <= node3);
        assert!(node1 < node3);
        assert!(node1 != node3);
        assert!(node1 <= node4);
        assert!(node1 < node4);
        assert!(node1 != node4);
        assert!(node3 <= node4);
        assert!(node3 < node4);
        assert!(node3 != node4);
    }
}
