use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{
    prelude::*,
    variable_type::{Numeric, OrderedContinuous},
};
use dypdl_heuristic_search::search_algorithm::{util::TimeKeeper, Apps};
use std::fmt::{Debug, Display};
use std::str::FromStr;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main_with_cost_type<T>(model: Model, config_filename: &str, time_keeper: &TimeKeeper)
where
    T: Numeric + Ord + Display + 'static,
    CostToDump: From<T>,
    <T as FromStr>::Err: Debug,
{
    let yaml = didp_mpi::read_config_yaml(config_filename);
    let map = yaml.as_hash().expect("Yaml file is not a hash");
    let parameters = didp_mpi::load_parameters_from_map::<T>(map);
    let f_evaluator_type = didp_mpi::load_f_evaluator_type_from_map(map);
    let progressive_parameters = didp_mpi::load_progressive_parameters_from_map(map);

    let (input, transition_evaluator, base_cost_evaluator) =
        didp_mpi::make_input_and_dual_bound_evaluators(
            model,
            f_evaluator_type,
            parameters.primal_bound,
        );

    println!("Time for initialization: {}s", time_keeper.elapsed_time());

    let mut solver = Apps::new(
        input,
        transition_evaluator,
        base_cost_evaluator,
        parameters,
        progressive_parameters,
    );

    let solution =
        didp_mpi::solve_and_dump_solutions(&mut solver, "history.csv", "solution.yaml").unwrap();

    didp_mpi::dump_solution(&solution);

    println!(
        "Time to the final solution: {}s",
        time_keeper.elapsed_time()
    );
}

fn main() {
    let time_keeper = TimeKeeper::default();

    let mut args = std::env::args();
    args.next();
    let model = didp_mpi::read_model(&mut args);
    let config_filename = args.next().expect("Config filename is not specified");

    match model.cost_type {
        CostType::Integer => main_with_cost_type::<Integer>(model, &config_filename, &time_keeper),
        CostType::Continuous => {
            main_with_cost_type::<OrderedContinuous>(model, &config_filename, &time_keeper)
        }
    }

    println!("Total time: {}s", time_keeper.elapsed_time());
}
