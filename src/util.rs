use crate::{
    bfs_node_with_distributed_id_chain::NodeGenerationResult,
    distributed_f_node::{DistributedFNode, FNodeEvaluators},
    distributed_f_node_message::DistributedFNodeMessage,
    distributed_id_chain::DistributedTransitionIdChain,
    mpi_anytime_search::MpiAnytimeSearchEvaluators,
};
use dypdl::{prelude::*, variable_type::Numeric};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::StateWithHashableSignatureVariables, FNode, SearchInput, StateInRegistry,
    StateRegistry, SuccessorGenerator, TransitionWithId,
};
use dypdl_heuristic_search::FEvaluatorType;
use std::rc::Rc;

pub fn make_input_and_dual_bound_evaluators<'a, T>(
    model: Model,
    f_evaluator_type: FEvaluatorType,
    primal_bound: Option<T>,
) -> (
    SearchInput<'a, FNode<T>>,
    impl FnMut(
        &FNode<T>,
        Rc<Transition>,
        &mut StateRegistry<T, FNode<T>>,
        Option<T>,
    ) -> Option<(Rc<FNode<T>>, bool)>,
    impl FnMut(T, T) -> T,
)
where
    T: Numeric + Ord,
{
    let model = Rc::new(model);
    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost: T, base_cost| f_evaluator_type.eval(cost, base_cost);
    let cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };
    let h_model = model.clone();
    let h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
    let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
    let node = FNode::<_>::generate_root_node(
        model.target.clone(),
        cost,
        &model,
        &h_evaluator,
        &f_evaluator,
        primal_bound,
    );
    let input = SearchInput {
        node,
        generator,
        solution_suffix: &[],
    };
    let transition_evaluator =
        move |node: &FNode<_>, transition, registry: &mut _, primal_bound| {
            node.insert_successor_node(
                transition,
                registry,
                &h_evaluator,
                &f_evaluator,
                primal_bound,
            )
        };

    (input, transition_evaluator, base_cost_evaluator)
}

pub fn make_input<'a, T>(
    model: Model,
    f_evaluator_type: FEvaluatorType,
    primal_bound: Option<T>,
) -> SearchInput<'a, DistributedFNodeMessage<T>, TransitionWithId, Rc<TransitionWithId>>
where
    T: Numeric + Ord,
{
    let model = Rc::new(model);
    let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);
    let cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };
    let h_model = model.clone();
    let remote_h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
    let remote_f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
    let node = DistributedFNodeMessage::generate_root_node(
        model.target.clone(),
        cost,
        &model,
        &remote_h_evaluator,
        &remote_f_evaluator,
        primal_bound,
    );

    SearchInput {
        node,
        generator,
        solution_suffix: &[],
    }
}

pub fn make_mpi_dual_bound_evaluators<T>(
    model: Rc<Model>,
    f_evaluator_type: FEvaluatorType,
) -> MpiAnytimeSearchEvaluators<
    impl FnMut(
        StateInRegistry,
        T,
        &TransitionWithId,
        &DistributedTransitionIdChain,
        &mut StateRegistry<T, DistributedFNode<T>>,
        Option<T>,
    ) -> NodeGenerationResult<Rc<DistributedFNode<T>>>,
    impl FnMut(
        StateWithHashableSignatureVariables,
        T,
        &TransitionWithId,
        &DistributedTransitionIdChain,
        Option<T>,
    ) -> Option<DistributedFNodeMessage<T>>,
    impl FnMut(T, T) -> T,
>
where
    T: Numeric + Ord,
{
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
    let h_model = model.clone();
    let remote_h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
    let remote_f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);

    let h_model = model.clone();
    let local_h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
    let local_f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
    let local_successor_evaluator =
        move |state, cost, transition: &_, chain: &_, registry: &mut _, primal_bound| {
            let evaluators = FNodeEvaluators {
                h: &local_h_evaluator,
                f: &local_f_evaluator,
            };
            DistributedFNode::insert_node(
                state,
                cost,
                transition,
                chain,
                registry,
                evaluators,
                primal_bound,
            )
        };
    let remote_successor_evaluator = move |state, cost, transition: &_, chain: &_, primal_bound| {
        let evaluators = FNodeEvaluators {
            h: &remote_h_evaluator,
            f: &remote_f_evaluator,
        };
        DistributedFNodeMessage::generate_node(
            state,
            cost,
            transition,
            chain,
            &model,
            evaluators,
            primal_bound,
        )
    };

    MpiAnytimeSearchEvaluators {
        local_successor_evaluator,
        remote_successor_evaluator,
        base_cost_evaluator,
    }
}

pub fn make_input_and_mpi_dual_bound_evaluators<'a, T>(
    model: Model,
    f_evaluator_type: FEvaluatorType,
    primal_bound: Option<T>,
) -> (
    SearchInput<'a, DistributedFNodeMessage<T>, TransitionWithId, Rc<TransitionWithId>>,
    MpiAnytimeSearchEvaluators<
        impl FnMut(
            StateInRegistry,
            T,
            &TransitionWithId,
            &DistributedTransitionIdChain,
            &mut StateRegistry<T, DistributedFNode<T>>,
            Option<T>,
        ) -> NodeGenerationResult<Rc<DistributedFNode<T>>>,
        impl FnMut(
            StateWithHashableSignatureVariables,
            T,
            &TransitionWithId,
            &DistributedTransitionIdChain,
            Option<T>,
        ) -> Option<DistributedFNodeMessage<T>>,
        impl FnMut(T, T) -> T,
    >,
)
where
    T: Numeric + Ord,
{
    let input = make_input(model, f_evaluator_type, primal_bound);
    let evaluators =
        make_mpi_dual_bound_evaluators(input.generator.model.clone(), f_evaluator_type);

    (input, evaluators)
}
