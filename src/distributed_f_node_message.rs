use crate::distributed_id_chain::GetDistributedTransitionIdChain;
use crate::node_data_type::NodeDatatype;
use crate::state_serializer::StateSerializer;

use super::distributed_f_node::DistributedFNode;
use super::distributed_id_chain::DistributedTransitionIdChain;
use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure::{
    exceed_bound, StateInformation, StateWithHashableSignatureVariables,
};
use dypdl_heuristic_search::search_algorithm::{StateInRegistry, TransitionWithId};
use mpi::datatype::DatatypeRef;
use mpi::{Address, Count};
use std::rc::Rc;

/// Node ordered by the f-value and associated with a transition ids chain
/// to be sent to another thread via message passing.
#[derive(Debug, Clone, PartialEq)]
pub struct DistributedFNodeMessage<T>
where
    T: Numeric,
{
    /// State.
    pub state: StateWithHashableSignatureVariables,
    /// g-value.
    pub g: T,
    /// h-value.
    pub h: T,
    /// f-value.
    pub f: T,
    /// Chain of transition ids to reach this node.
    pub transition_id_chain: DistributedTransitionIdChain,
}

impl<T> From<DistributedFNodeMessage<T>> for DistributedFNode<T>
where
    T: Numeric,
{
    fn from(node: DistributedFNodeMessage<T>) -> Self {
        Self::new(
            StateInRegistry::from(node.state),
            node.g,
            node.h,
            node.f,
            Rc::new(node.transition_id_chain),
        )
    }
}

impl<T> NodeDatatype for DistributedFNodeMessage<T>
where
    T: Numeric,
{
    fn get_total_size(serializer: &StateSerializer) -> usize {
        serializer.get_total_size() + DistributedTransitionIdChain::get_total_size()
    }

    fn get_datatype_blocklengths(serializer: &StateSerializer) -> Vec<Count> {
        let mut blocklengths = Vec::from(serializer.get_datatype_blocklengths());
        blocklengths.extend(DistributedTransitionIdChain::get_datatype_blocklengths());

        blocklengths
    }

    fn get_datatype_displacement(serializer: &StateSerializer) -> Vec<Address> {
        let mut displacement = Vec::from(serializer.get_datatype_displacement());
        let offset = serializer.get_total_size();
        displacement.extend(
            DistributedTransitionIdChain::get_datatype_displacements()
                .into_iter()
                .map(|x| x + offset as Address),
        );

        displacement
    }

    fn get_datatype_types(serializer: &StateSerializer) -> Vec<DatatypeRef<'static>> {
        let mut types = Vec::from(serializer.get_datatype_types());
        types.extend(DistributedTransitionIdChain::get_datatype_types());

        types
    }

    fn serialize_to(&self, serializer: &StateSerializer, buffer: &mut [u8]) {
        let offset = serializer.get_total_size();
        serializer.serialize_to(&self.state, self.g, self.h, self.f, &mut buffer[..offset]);
        self.transition_id_chain.serialize_to(&mut buffer[offset..]);
    }

    fn deserialize(serializer: &StateSerializer, buffer: &[u8]) -> Self {
        let offset = serializer.get_total_size();
        let (state, g, h, f) = serializer.deserialize(&buffer[..offset]);
        let transition_id_chain = DistributedTransitionIdChain::deserialize(&buffer[offset..]);

        Self {
            state,
            g,
            h,
            f,
            transition_id_chain,
        }
    }
}

impl<T> DistributedFNodeMessage<T>
where
    T: Numeric,
{
    pub fn generate_root_node<S, H, F>(
        state: S,
        cost: T,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        StateWithHashableSignatureVariables: From<S>,
        H: FnOnce(&StateWithHashableSignatureVariables) -> Option<T>,
        F: FnOnce(T, T, &StateWithHashableSignatureVariables) -> T,
    {
        let state = StateWithHashableSignatureVariables::from(state);
        let h = h_evaluator(&state)?;
        let f = f_evaluator(cost, h, &state);

        if exceed_bound(model, f, primal_bound) {
            return None;
        }

        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };

        Some(Self {
            state,
            g: cost,
            h,
            f,
            transition_id_chain: DistributedTransitionIdChain::default(),
        })
    }

    pub fn bound(&self, model: &Model) -> T {
        if model.reduce_function == ReduceFunction::Min {
            -self.f
        } else {
            self.f
        }
    }
}

impl<T> DistributedFNode<T>
where
    T: Numeric,
{
    /// Generates a successor node as a sendable message.
    pub fn generate_sendable_successor_node<V, H, F>(
        &self,
        transition: &TransitionWithId<V>,
        model: &Model,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<DistributedFNodeMessage<T>>
    where
        V: TransitionInterface,
        H: FnOnce(&StateWithHashableSignatureVariables) -> Option<T>,
        F: FnOnce(T, T, &StateWithHashableSignatureVariables) -> T,
    {
        let (state, g) =
            model.generate_successor_state(self.state(), self.cost(model), transition, None)?;
        let h = h_evaluator(&state)?;
        let f = f_evaluator(g, h, &state);

        if exceed_bound(model, f, primal_bound) {
            return None;
        }

        let (h, f) = if model.reduce_function == ReduceFunction::Max {
            (h, f)
        } else {
            (-h, -f)
        };

        let transition_id_chain = self
            .get_distributed_transition_id_chain()
            .generate_successor(transition.id, transition.forced);

        Some(DistributedFNodeMessage {
            state,
            g,
            h,
            f,
            transition_id_chain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_root_message_some_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let expected_state = StateWithHashableSignatureVariables::from(state.clone());
        let cost = 0;
        let h_evaluator = |_: &StateWithHashableSignatureVariables| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateWithHashableSignatureVariables| g + h;
        let primal_bound = None;

        let node = DistributedFNodeMessage::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.state, expected_state);
        assert_eq!(node.g, 0);
        assert_eq!(node.h, 0);
        assert_eq!(node.f, 0);
        assert_eq!(
            node.transition_id_chain,
            DistributedTransitionIdChain::default()
        );
    }

    #[test]
    fn generate_root_message_some_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let expected_state = StateWithHashableSignatureVariables::from(state.clone());
        let cost = 0;
        let h_evaluator = |_: &StateWithHashableSignatureVariables| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateWithHashableSignatureVariables| g + h;
        let primal_bound = None;

        let node = DistributedFNodeMessage::generate_root_node(
            state,
            cost,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.state, expected_state);
        assert_eq!(node.g, 0);
        assert_eq!(node.h, 0);
        assert_eq!(node.f, 0);
        assert_eq!(
            node.transition_id_chain,
            DistributedTransitionIdChain::default()
        );
    }

    #[test]
    fn generate_root_message_pruned_by_bound_min() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let cost = 0;
        let h_evaluator = |_: &StateWithHashableSignatureVariables| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateWithHashableSignatureVariables| g + h;
        let primal_bound = Some(-1);

        let node = DistributedFNodeMessage::generate_root_node(
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
    fn generate_root_message_pruned_by_bound_max() {
        let mut model = dypdl::Model::default();
        model.set_maximize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let cost = 0;
        let h_evaluator = |_: &StateWithHashableSignatureVariables| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &StateWithHashableSignatureVariables| g + h;
        let primal_bound = Some(1);

        let node = DistributedFNodeMessage::generate_root_node(
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
    fn generate_root_message_pruned_by_h() {
        let mut model = dypdl::Model::default();
        model.set_minimize();
        let variable = model.add_integer_variable("variable", 0);
        assert!(variable.is_ok());
        let state = model.target.clone();
        let cost = 0;
        let h_evaluator = |_: &StateWithHashableSignatureVariables| None;
        let f_evaluator = |g: i32, h: i32, _: &StateWithHashableSignatureVariables| g + h;
        let primal_bound = None;

        let node = DistributedFNodeMessage::generate_root_node(
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
    fn generate_sendable_successor_some_min() {
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
        let expected_state = transition.apply(&state, &model.table_registry);
        let cost = 1;
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
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
        node.get_distributed_transition_id_chain().id.set(Some(0));
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
        let successor = node.generate_sendable_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_some());
        let successor = successor.unwrap();
        assert_eq!(successor.state, expected_state);
        assert_eq!(successor.g, 2);
        assert_eq!(successor.h, 0);
        assert_eq!(successor.f, -2);
    }

    #[test]
    fn generate_sendable_successor_some_max() {
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
        let expected_state = transition.apply(&state, &model.table_registry);
        let cost = 1;
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
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
        node.get_distributed_transition_id_chain().id.set(Some(0));
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
        let successor = node.generate_sendable_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_some());
        let successor = successor.unwrap();
        assert_eq!(successor.state, expected_state);
        assert_eq!(successor.g, 2);
        assert_eq!(successor.h, 0);
        assert_eq!(successor.f, 2);
    }

    #[test]
    fn generate_sendable_successor_pruned_by_constraint() {
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
        node.get_distributed_transition_id_chain().id.set(Some(0));
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
        let successor = node.generate_sendable_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_none());
    }

    #[test]
    fn generate_sendable_successor_pruned_by_bound_min() {
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

        let primal_bound = Some(2);
        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;

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
        node.get_distributed_transition_id_chain().id.set(Some(0));

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
        let successor = node.generate_sendable_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_none());
    }

    #[test]
    fn generate_sendable_successor_pruned_by_bound_max() {
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

        let primal_bound = Some(3);
        let h_evaluator = |_: &_| Some(3);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;

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
        node.get_distributed_transition_id_chain().id.set(Some(0));

        let h_evaluator = |_: &_| Some(1);
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
        let successor = node.generate_sendable_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_none());
    }

    #[test]
    fn generate_sendable_successor_pruned_by_h() {
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
        node.get_distributed_transition_id_chain().id.set(Some(0));
        let h_evaluator = |_: &_| None;
        let f_evaluator = |g: i32, h: i32, _: &_| g + h;
        let successor = node.generate_sendable_successor_node(
            &transition,
            &model,
            h_evaluator,
            f_evaluator,
            primal_bound,
        );

        assert!(successor.is_none());
    }
}
