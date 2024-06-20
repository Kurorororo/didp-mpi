use crate::distributed_id_chain::GetDistributedTransitionIdChain;
use crate::is_float::IsFloat;
use crate::node_data_type::NodeDatatype;
use crate::state_serializer::StateSerializer;
use crate::{distributed_f_node::FNodeEvaluators, node_message::NodeMessage};

use super::distributed_f_node::DistributedFNode;
use super::distributed_id_chain::DistributedTransitionIdChain;
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::{
        self, HashableSignatureVariables, StateInformation, StateWithHashableSignatureVariables,
    },
    StateInRegistry, TransitionWithId,
};
use mpi::{datatype::DatatypeRef, traits::*, Address, Count, Rank};
use std::mem;
use std::rc::Rc;
use zerocopy::{AsBytes, FromBytes};

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

impl<T> From<DistributedFNode<T>> for DistributedFNodeMessage<T>
where
    T: Numeric,
{
    fn from(node: DistributedFNode<T>) -> Self {
        Self {
            state: StateWithHashableSignatureVariables::from(node.state().clone()),
            g: node.g,
            h: node.h,
            f: node.f,
            transition_id_chain: (*node.get_distributed_transition_id_chain()).clone(),
        }
    }
}

impl<T> GetDistributedTransitionIdChain for DistributedFNodeMessage<T>
where
    T: Numeric,
{
    fn get_distributed_transition_id_chain(&self) -> &DistributedTransitionIdChain {
        &self.transition_id_chain
    }
}

impl<T> NodeDatatype<T> for DistributedFNodeMessage<T>
where
    T: Numeric + IsFloat,
{
    type S = Self;

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

    fn deserialize(serializer: &StateSerializer, buffer: &[u8]) -> Self {
        let mut offset = serializer.get_total_size();
        let state = serializer.deserialize(&buffer[..offset]);

        let (g, h, f) = if T::is_float() {
            let size = mem::size_of::<Continuous>();
            let g = T::from(Continuous::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = mem::size_of::<Continuous>();
            let h = T::from(Continuous::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = mem::size_of::<Continuous>();
            let f = T::from(Continuous::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            (g, h, f)
        } else {
            let size = mem::size_of::<Integer>();
            let g = T::from(Integer::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = mem::size_of::<Integer>();
            let h = T::from(Integer::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = mem::size_of::<Integer>();
            let f = T::from(Integer::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            (g, h, f)
        };

        let transition_id_chain = DistributedTransitionIdChain::deserialize(&buffer[offset..]);

        Self {
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
            T::from(Continuous::read_from(&data[offset..offset + size]).unwrap())
        } else {
            let size = mem::size_of::<Integer>();
            let offset = serializer.get_total_size() + 2 * size;
            T::from(Integer::read_from(&data[offset..offset + size]).unwrap())
        };

        let bound = if model.reduce_function == ReduceFunction::Min {
            -bound
        } else {
            bound
        };

        Some(bound)
    }
}

impl<T: IsFloat> NodeMessage<T> for DistributedFNodeMessage<T> {
    fn state(&self) -> &StateWithHashableSignatureVariables {
        &self.state
    }

    fn signature(&self) -> &HashableSignatureVariables {
        &self.state.signature_variables
    }

    fn cost(&self, _: &Model) -> T {
        self.g
    }

    fn bound(&self, model: &Model) -> Option<T> {
        if model.reduce_function == ReduceFunction::Max {
            Some(self.f)
        } else {
            Some(-self.f)
        }
    }

    fn set_parent_rank(&self, parent_rank: Rank) {
        self.transition_id_chain.parent_rank.set(Some(parent_rank));
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

        if data_structure::exceed_bound(model, f, primal_bound) {
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

    /// Generates a successor node as a sendable message.
    pub fn generate_node<V, H, F>(
        state: StateWithHashableSignatureVariables,
        g: T,
        transition: &TransitionWithId<V>,
        transition_id_chain: &DistributedTransitionIdChain,
        model: &Model,
        evaluators: FNodeEvaluators<H, F>,
        primal_bound: Option<T>,
    ) -> Option<Self>
    where
        V: TransitionInterface,
        H: FnOnce(&StateWithHashableSignatureVariables) -> Option<T>,
        F: FnOnce(T, T, &StateWithHashableSignatureVariables) -> T,
    {
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
            transition_id_chain.generate_successor(transition.id, transition.forced);

        Some(Self {
            state,
            g,
            h,
            f,
            transition_id_chain,
        })
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

        let evaluators = FNodeEvaluators {
            h: h_evaluator,
            f: f_evaluator,
        };

        DistributedFNodeMessage::generate_node(
            state,
            g,
            transition,
            self.get_distributed_transition_id_chain(),
            model,
            evaluators,
            primal_bound,
        )
    }
}

#[cfg(test)]
mod tests {
    use dypdl::variable_type::OrderedContinuous;

    use super::*;

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

        let node = DistributedFNodeMessage {
            state: StateWithHashableSignatureVariables::from(state),
            g: 19,
            h: -23,
            f: -42,
            transition_id_chain: successor,
        };

        let serializer = StateSerializer::with_model(&model);

        let mut buffer =
            vec![0u8; DistributedFNodeMessage::<Integer>::get_total_size(&serializer,)];

        node.serialize_to(&serializer, &mut buffer);
        let deserialized = DistributedFNodeMessage::<Integer>::deserialize(&serializer, &buffer);

        assert_eq!(
            DistributedFNodeMessage::<Integer>::get_bound_from_buffer(&model, &serializer, &buffer),
            Some(42)
        );
        assert_eq!(deserialized, node);
    }

    #[test]
    fn test_serialize_continuous() {
        let (model, state) = create_model_and_state();
        let chain = DistributedTransitionIdChain::default();
        chain.id.set(Some(0));
        let successor = chain.generate_successor(0, false);
        successor.parent_rank.set(Some(1));

        let node = DistributedFNodeMessage {
            state: StateWithHashableSignatureVariables::from(state),
            g: OrderedContinuous::from(1.9),
            h: OrderedContinuous::from(-2.3),
            f: OrderedContinuous::from(-4.2),
            transition_id_chain: successor,
        };

        let serializer = StateSerializer::with_model(&model);

        let mut buffer =
            vec![0u8; DistributedFNodeMessage::<OrderedContinuous>::get_total_size(&serializer)];

        node.serialize_to(&serializer, &mut buffer);
        let deserialized =
            DistributedFNodeMessage::<OrderedContinuous>::deserialize(&serializer, &buffer);

        assert_eq!(
            DistributedFNodeMessage::<OrderedContinuous>::get_bound_from_buffer(
                &model,
                &serializer,
                &buffer
            ),
            Some(OrderedContinuous::from(4.2))
        );
        assert_eq!(deserialized, node);
    }

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
        let successor: Option<DistributedFNodeMessage<i32>> = node
            .generate_sendable_successor_node(
                &transition,
                &model,
                h_evaluator,
                f_evaluator,
                primal_bound,
            );

        assert!(successor.is_none());
    }
}
