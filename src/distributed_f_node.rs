use crate::bfs_node_with_distributed_id_chain::NodeGenerationResult;
use crate::distributed_f_node_message::DistributedFNodeMessage;
use crate::distributed_id_chain::GeRcDistributedTransitionIdChain;
use crate::is_float::IsFloat;
use crate::node_data_type::NodeDatatype;
use crate::state_serializer::StateSerializer;
use crate::{
    bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain,
    distributed_id_chain::GetDistributedTransitionIdChain,
};

use super::distributed_id_chain::DistributedTransitionIdChain;
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::{self, StateInformation},
    StateInRegistry, StateRegistry, TransitionWithId,
};
use mpi::{datatype::DatatypeRef, traits::*, Address, Count};
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::Display;
use std::mem;
use std::rc::Rc;
use zerocopy::{FromBytes, IntoBytes};

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

/// Evaluators for FNode.
#[derive(Clone)]
pub struct FNodeEvaluators<H, F> {
    /// h.
    pub h: H,
    /// f.
    pub f: F,
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

    /// Generate a successor node given a state and the g-value.
    pub fn generate_node<S, V, H, F>(
        state: S,
        g: T,
        transition: &TransitionWithId<V>,
        transition_id_chain: &DistributedTransitionIdChain,
        model: &Model,
        evaluators: FNodeEvaluators<H, F>,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        S: StateInterface,
        StateInRegistry: From<S>,
        V: TransitionInterface,
        H: FnOnce(&StateInRegistry) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry) -> T,
    {
        let state = StateInRegistry::from(state);
        let h = (evaluators.h)(&state)?;
        let f = (evaluators.f)(g, h, &state);

        if data_structure::exceed_bound(model, f, primal_bound) {
            return None;
        }

        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };

        let transition_id_chain =
            Rc::new(transition_id_chain.generate_successor(transition.id, transition.forced));

        Some(Self::new(state, g, h, f, transition_id_chain))
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
        let (state, g): (StateInRegistry, _) =
            model.generate_successor_state(&self.state, self.g, transition, None)?;
        let evaluators = FNodeEvaluators {
            h: h_evaluator,
            f: f_evaluator,
        };

        Self::generate_node(
            state,
            g,
            transition,
            &self.transition_id_chain,
            model,
            evaluators,
            primal_bound,
        )
    }

    /// Inserts a successor node generated from a given state and the g-value into the registry.
    pub fn insert_node<V, H, F>(
        state: StateInRegistry,
        g: T,
        transition: &TransitionWithId<V>,
        transition_id_chain: &DistributedTransitionIdChain,
        registry: &mut StateRegistry<T, Self>,
        evaluators: FNodeEvaluators<H, F>,
        primal_bound: Option<T>,
    ) -> NodeGenerationResult<Rc<Self>>
    where
        V: TransitionInterface,
        H: FnOnce(&StateInRegistry) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry) -> T,
    {
        let model = registry.model().clone();
        let maximize = model.reduce_function == ReduceFunction::Max;
        let mut is_pruned_by_bound = false;

        let constructor = |state, g, other: Option<&Self>| {
            let h = if let Some(other) = other {
                if maximize {
                    other.h
                } else {
                    -other.h
                }
            } else if let Some(h) = (evaluators.h)(&state) {
                h
            } else {
                is_pruned_by_bound = true;

                return None;
            };
            let f = (evaluators.f)(g, h, &state);

            if data_structure::exceed_bound(&model, f, primal_bound) {
                is_pruned_by_bound = true;

                return None;
            }

            let (h, f) = if maximize { (h, f) } else { (-h, -f) };

            let transition_id_chain =
                Rc::new(transition_id_chain.generate_successor(transition.id, transition.forced));

            Some(Self::new(state, g, h, f, transition_id_chain))
        };

        let result = registry.insert_with(state, g, constructor);

        let mut n_dominated_before_closed = 0;
        let mut n_dominated_after_closed = 0;

        for d in result.dominated.iter() {
            if d.is_closed() {
                n_dominated_after_closed += 1;
            } else {
                n_dominated_before_closed += 1;
                d.close();
            }
        }

        NodeGenerationResult {
            node: result.information,
            is_pruned_by_bound,
            dominated_before_closed: n_dominated_before_closed,
            dominated_after_closed: n_dominated_after_closed,
        }
    }

    /// Inserts a successor node into the registry.
    pub fn insert_successor_node<V, H, F>(
        &self,
        transition: &TransitionWithId<V>,
        registry: &mut StateRegistry<T, Self>,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> NodeGenerationResult<Rc<Self>>
    where
        V: TransitionInterface,
        H: FnOnce(&StateInRegistry) -> Option<T>,
        F: FnOnce(T, T, &StateInRegistry) -> T,
    {
        if let Some((state, g)) =
            registry
                .model()
                .generate_successor_state(&self.state, self.g, transition, None)
        {
            let evaluators = FNodeEvaluators {
                h: h_evaluator,
                f: f_evaluator,
            };

            Self::insert_node(
                state,
                g,
                transition,
                &self.transition_id_chain,
                registry,
                evaluators,
                primal_bound,
            )
        } else {
            NodeGenerationResult::default()
        }
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
    fn get_distributed_transition_id_chain(&self) -> &DistributedTransitionIdChain {
        &self.transition_id_chain
    }
}

impl<T> GeRcDistributedTransitionIdChain for DistributedFNode<T>
where
    T: Numeric,
{
    #[inline]
    fn get_rc_distributed_transition_id_chain(&self) -> &Rc<DistributedTransitionIdChain> {
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

impl<T> NodeDatatype<T> for DistributedFNode<T>
where
    T: IsFloat,
{
    type S = DistributedFNodeMessage<T>;

    fn get_total_size(serializer: &StateSerializer) -> usize {
        let size = if T::is_float() {
            mem::size_of::<Continuous>()
        } else {
            mem::size_of::<Integer>()
        };

        serializer.get_total_size() + 3 * size + DistributedTransitionIdChain::get_total_size()
    }

    fn get_datatype_blocklengths(serializer: &StateSerializer) -> Vec<Count> {
        let mut blocklengths = Vec::from(serializer.get_datatype_blocklengths());
        blocklengths.push(3);
        blocklengths.extend(DistributedTransitionIdChain::get_datatype_blocklengths());

        blocklengths
    }

    fn get_datatype_displacements(serializer: &StateSerializer) -> Vec<Address> {
        let mut displacements = Vec::from(serializer.get_datatype_displacements());
        let mut offset = serializer.get_total_size();
        displacements.push(offset as Address);

        if T::is_float() {
            offset += 3 * mem::size_of::<Continuous>();
        } else {
            offset += 3 * mem::size_of::<Integer>();
        };

        displacements.extend(
            DistributedTransitionIdChain::get_datatype_displacements()
                .into_iter()
                .map(|x| x + offset as Address),
        );

        displacements
    }

    fn get_datatype_types(serializer: &StateSerializer) -> Vec<DatatypeRef<'static>> {
        let mut types = Vec::from(serializer.get_datatype_types());

        if T::is_float() {
            types.push(Continuous::equivalent_datatype());
        } else {
            types.push(Integer::equivalent_datatype());
        }

        types.extend(DistributedTransitionIdChain::get_datatype_types());

        types
    }

    fn serialize_to(&self, serializer: &StateSerializer, buffer: &mut [u8]) {
        let mut offset = serializer.get_total_size();
        serializer.serialize_to(&self.state, &mut buffer[..offset]);

        if T::is_float() {
            let size = mem::size_of::<Continuous>();
            buffer[offset..offset + size].copy_from_slice(self.g.to_continuous().as_bytes());
            offset += size;
            buffer[offset..offset + size].copy_from_slice(self.h.to_continuous().as_bytes());
            offset += size;
            buffer[offset..offset + size].copy_from_slice(self.f.to_continuous().as_bytes());
            offset += size;
        } else {
            let size = mem::size_of::<Integer>();
            buffer[offset..offset + size].copy_from_slice(self.g.to_integer().as_bytes());
            offset += size;
            buffer[offset..offset + size].copy_from_slice(self.h.to_integer().as_bytes());
            offset += size;
            buffer[offset..offset + size].copy_from_slice(self.f.to_integer().as_bytes());
            offset += size;
        };

        self.transition_id_chain.serialize_to(&mut buffer[offset..]);
    }

    fn deserialize(serializer: &StateSerializer, buffer: &[u8]) -> DistributedFNodeMessage<T> {
        let mut offset = serializer.get_total_size();
        let state = serializer.deserialize(&buffer[..offset]);

        let (g, h, f) = if T::is_float() {
            let size = mem::size_of::<Continuous>();
            let g = T::from(Continuous::read_from_bytes(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = mem::size_of::<Continuous>();
            let h = T::from(Continuous::read_from_bytes(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = mem::size_of::<Continuous>();
            let f = T::from(Continuous::read_from_bytes(&buffer[offset..offset + size]).unwrap());
            offset += size;

            (g, h, f)
        } else {
            let size = mem::size_of::<Integer>();
            let g = T::from(Integer::read_from_bytes(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = mem::size_of::<Integer>();
            let h = T::from(Integer::read_from_bytes(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = mem::size_of::<Integer>();
            let f = T::from(Integer::read_from_bytes(&buffer[offset..offset + size]).unwrap());
            offset += size;

            (g, h, f)
        };

        let transition_id_chain = DistributedTransitionIdChain::deserialize(&buffer[offset..]);

        DistributedFNodeMessage {
            state,
            g,
            h,
            f,
            transition_id_chain,
        }
    }

    fn get_bound_from_buffer(
        model: &Model,
        serializer: &StateSerializer,
        data: &[u8],
    ) -> Option<T> {
        let bound = if T::is_float() {
            let size = mem::size_of::<Continuous>();
            let offset = serializer.get_total_size() + 2 * size;
            T::from(Continuous::read_from_bytes(&data[offset..offset + size]).unwrap())
        } else {
            let size = mem::size_of::<Integer>();
            let offset = serializer.get_total_size() + 2 * size;
            T::from(Integer::read_from_bytes(&data[offset..offset + size]).unwrap())
        };

        let bound = if model.reduce_function == ReduceFunction::Min {
            -bound
        } else {
            bound
        };

        Some(bound)
    }
}

#[cfg(test)]
mod tests {
    use data_structure::StateWithHashableSignatureVariables;
    use dypdl::variable_type::OrderedContinuous;

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

        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_some());
        let successor = result.node.unwrap();
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
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_some());
        let successor = result.node.unwrap();
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
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_none());
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
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_some());
        let successor = result.node.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());

        let expected_state: StateInRegistry = transition2.apply(&state, &model.table_registry);
        let result = node.insert_successor_node(
            &transition2,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 1);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_some());
        let successor = result.node.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
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
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_some());
        let successor = result.node.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());

        let expected_state: StateInRegistry = transition2.apply(&state, &model.table_registry);
        let result = node.insert_successor_node(
            &transition2,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 1);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_some());
        let successor = result.node.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), Some(2));
        assert!(!successor.is_closed());
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
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_some());
        let successor = result.node.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 3);
        assert_eq!(successor.bound(&model), Some(3));
        assert!(!successor.is_closed());

        let result = node.insert_successor_node(
            &transition1,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_none());
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
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_some());
        let successor = result.node.unwrap();
        assert_eq!(successor.state(), &expected_state);
        assert_eq!(successor.cost(&model), 3);
        assert_eq!(successor.bound(&model), Some(3));
        assert!(!successor.is_closed());

        let result = node.insert_successor_node(
            &transition1,
            &mut registry,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );
        assert!(!result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_none());
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
        assert!(result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_none());
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
        transition1.set_cost(IntegerExpression::Cost - 1);
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
        assert!(result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_none());
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
        assert!(result.is_pruned_by_bound);
        assert_eq!(result.dominated_before_closed, 0);
        assert_eq!(result.dominated_after_closed, 0);
        assert!(result.node.is_none());
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
        assert!(node3.node.is_some());
        let node3 = node3.node.unwrap();

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
        assert!(node3.node.is_some());
        let node3 = node3.node.unwrap();

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

    fn create_model_and_state() -> (Model, State) {
        let mut model = Model::default();

        let object_type_1 = model.add_object_type("object1", 4);
        assert!(object_type_1.is_ok());
        let object_type_1 = object_type_1.unwrap();

        let object_type_2 = model.add_object_type("object2", 10);
        assert!(object_type_2.is_ok());
        let object_type_2 = object_type_2.unwrap();

        let object_type_3 = model.add_object_type("object3", 7);
        assert!(object_type_3.is_ok());
        let object_type_3 = object_type_3.unwrap();

        let set1 = Set::with_capacity(4);
        let v = model.add_set_variable("set1", object_type_1, set1);
        assert!(v.is_ok());

        let mut set2 = Set::with_capacity(10);
        set2.set_range(1..3, true);
        set2.set_range(7..10, true);
        let v = model.add_set_variable("set2", object_type_2, set2);
        assert!(v.is_ok());

        let mut set3 = Set::with_capacity(7);
        set3.set_range(0..7, true);
        let v = model.add_set_variable("set3", object_type_3, set3);
        assert!(v.is_ok());

        let v = model.add_element_variable("element1", object_type_2, 5);
        assert!(v.is_ok());

        let v = model.add_element_variable("element2", object_type_3, 1);
        assert!(v.is_ok());

        let v = model.add_integer_variable("integer1", 0);
        assert!(v.is_ok());

        let v = model.add_integer_variable("integer2", Integer::MAX);
        assert!(v.is_ok());

        let v = model.add_integer_variable("integer3", 10);
        assert!(v.is_ok());

        let v = model.add_integer_variable("integer4", Integer::MIN);
        assert!(v.is_ok());

        let v = model.add_continuous_variable("continuous1", Continuous::MIN);
        assert!(v.is_ok());

        let v = model.add_continuous_variable("continuous2", std::f64::consts::PI);
        assert!(v.is_ok());

        let v = model.add_continuous_variable("continuous4", Continuous::MAX);
        assert!(v.is_ok());

        let v = model.add_element_resource_variable("element_resource1", object_type_3, true, 9);
        assert!(v.is_ok());

        let v = model.add_element_resource_variable("element_resource2", object_type_1, false, 0);
        assert!(v.is_ok());

        let v = model.add_element_resource_variable("element_resource3", object_type_2, true, 4);
        assert!(v.is_ok());

        let v = model.add_integer_resource_variable("integer_resource2", false, -50);
        assert!(v.is_ok());

        let v = model.add_integer_resource_variable("integer_resource3", false, Integer::MIN);
        assert!(v.is_ok());

        let v = model.add_continuous_resource_variable("continuous_resource1", true, 0.0);
        assert!(v.is_ok());

        let state = model.target.clone();

        (model, state)
    }

    #[test]
    fn test_serialize_integer() {
        let (model, state) = create_model_and_state();
        let chain = DistributedTransitionIdChain::default();
        chain.id.set(Some(0));
        let successor = chain.generate_successor(0, false);
        successor.parent_rank.set(Some(1));

        let node = DistributedFNode {
            state: StateInRegistry::from(state.clone()),
            g: 19,
            h: -23,
            f: -42,
            transition_id_chain: Rc::new(successor.clone()),
            closed: Cell::new(false),
        };

        let serializer = StateSerializer::with_model(&model);

        let mut buffer =
            vec![0u8; DistributedFNodeMessage::<Integer>::get_total_size(&serializer,)];

        node.serialize_to(&serializer, &mut buffer);
        let deserialized = DistributedFNodeMessage::<Integer>::deserialize(&serializer, &buffer);

        let expected = DistributedFNodeMessage {
            state: StateWithHashableSignatureVariables::from(state),
            g: 19,
            h: -23,
            f: -42,
            transition_id_chain: successor,
        };

        assert_eq!(
            DistributedFNode::<Integer>::get_bound_from_buffer(&model, &serializer, &buffer),
            Some(42)
        );
        assert_eq!(deserialized, expected);
    }

    #[test]
    fn test_serialize_continuous() {
        let (model, state) = create_model_and_state();
        let chain = DistributedTransitionIdChain::default();
        chain.id.set(Some(0));
        let successor = chain.generate_successor(0, false);
        successor.parent_rank.set(Some(1));

        let node = DistributedFNode {
            state: StateInRegistry::from(state.clone()),
            g: OrderedContinuous::from(1.9),
            h: OrderedContinuous::from(-2.3),
            f: OrderedContinuous::from(-4.2),
            transition_id_chain: Rc::new(successor.clone()),
            closed: Cell::new(false),
        };

        let serializer = StateSerializer::with_model(&model);

        let mut buffer =
            vec![0u8; DistributedFNodeMessage::<OrderedContinuous>::get_total_size(&serializer)];

        node.serialize_to(&serializer, &mut buffer);
        let deserialized =
            DistributedFNodeMessage::<OrderedContinuous>::deserialize(&serializer, &buffer);

        let expected = DistributedFNodeMessage {
            state: StateWithHashableSignatureVariables::from(state),
            g: OrderedContinuous::from(1.9),
            h: OrderedContinuous::from(-2.3),
            f: OrderedContinuous::from(-4.2),
            transition_id_chain: successor,
        };

        assert_eq!(
            DistributedFNode::<OrderedContinuous>::get_bound_from_buffer(
                &model,
                &serializer,
                &buffer
            ),
            Some(OrderedContinuous::from(4.2))
        );
        assert_eq!(deserialized, expected);
    }
}
