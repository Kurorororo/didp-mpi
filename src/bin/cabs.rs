use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{
    prelude::*,
    variable_type::{Numeric, OrderedContinuous},
};
use dypdl_heuristic_search::{
    search_algorithm::{
        beam_search, util::TimeKeeper, Cabs, FNode, SearchInput, SuccessorGenerator,
    },
    FEvaluatorType,
};
use std::str::FromStr;
use std::{
    fmt::{Debug, Display},
    rc::Rc,
};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main_with_cost_type<T>(model: Model, config_filename: &str)
where
    T: Numeric + Ord + Display + 'static,
    CostToDump: From<T>,
    <T as FromStr>::Err: Debug,
{
    let yaml = didp_mpi::read_config_yaml(config_filename);
    let map = yaml.as_hash().expect("Yaml file is not a hash");
    let parameters = didp_mpi::load_cabs_parameters_from_map::<T>(map);
    let f_evaluator_type = didp_mpi::load_f_evaluator_type_from_map(map);
    let model = Rc::new(model);

    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
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
    let node = FNode::generate_root_node(
        model.target.clone(),
        cost,
        &model,
        &h_evaluator,
        &f_evaluator,
        parameters.beam_search_parameters.parameters.primal_bound,
    );
    let input = SearchInput {
        node,
        generator,
        solution_suffix: &[],
    };
    let transition_evaluator = move |node: &FNode<_>, transition, primal_bound| {
        node.generate_successor_node(transition, &model, &h_evaluator, &f_evaluator, primal_bound)
    };
    let beam_search = move |input: &SearchInput<_, _>, parameters| {
        beam_search(
            input,
            &transition_evaluator,
            base_cost_evaluator,
            parameters,
        )
    };
    let mut solver = Cabs::<_, FNode<_>, _>::new(input, beam_search, parameters);

    let solution =
        didp_mpi::solve_and_dump_solutions(&mut solver, "history.csv", "solution.yaml").unwrap();

    didp_mpi::dump_solution(&solution);
}

fn main() {
    let time_keeper = TimeKeeper::default();

    let mut args = std::env::args();
    args.next();
    let model = didp_mpi::read_model(&mut args);
    let config_filename = args.next().expect("Config filename is not specified");

    match model.cost_type {
        CostType::Integer => main_with_cost_type::<Integer>(model, &config_filename),
        CostType::Continuous => main_with_cost_type::<OrderedContinuous>(model, &config_filename),
    }

    println!("Total time: {}s", time_keeper.elapsed_time());
}
