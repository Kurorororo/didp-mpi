use dypdl::prelude::*;
use dypdl_heuristic_search::search_algorithm::data_structure::HashableSignatureVariables;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

pub fn create_fx_hash() -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u32 = 0x5583c24d;

    |signature: &HashableSignatureVariables| -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write_u32(SEED);
        signature.hash(&mut hasher);
        hasher.finish()
    }
}

fn create_random_table(model: &Model, seed: u64) -> Vec<Vec<u64>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let n = model.state_metadata.number_of_set_variables();
    let mut random_table = Vec::with_capacity(n);

    for i in 0..n {
        let object_id = model.state_metadata.set_variable_to_object[i];
        let m = model.state_metadata.object_numbers[object_id];
        let random_row = (0..m).map(|_| rng.gen::<u64>()).collect::<Vec<_>>();
        random_table.push(random_row);
    }

    random_table
}

fn create_abstracted_random_table(
    model: &Model,
    seed: u64,
    zero_probability: f64,
) -> Vec<Vec<u64>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let n = model.state_metadata.number_of_set_variables();
    let mut random_table = Vec::with_capacity(n);

    for i in 0..n {
        let object_id = model.state_metadata.set_variable_to_object[i];
        let m = model.state_metadata.object_numbers[object_id];
        let random_row = (0..m)
            .map(|_| {
                if rng.gen_bool(zero_probability) {
                    0
                } else {
                    rng.gen::<u64>()
                }
            })
            .collect::<Vec<_>>();
        random_table.push(random_row);
    }

    random_table
}

fn set_zobrist_hash(random_table: &[Vec<u64>], set_variables: &[Set]) -> u64 {
    let mut hash_value = 0;

    for (v, row) in set_variables.iter().zip(random_table.iter()) {
        for i in v.ones() {
            hash_value ^= row[i];
        }
    }

    hash_value
}

pub fn create_set_zobrist_hash(
    model: &Model,
    zero_probability: Option<f64>,
) -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u64 = 42;

    let random_table = if let Some(p) = zero_probability {
        create_abstracted_random_table(model, SEED, p)
    } else {
        create_random_table(model, SEED)
    };

    move |signature: &HashableSignatureVariables| -> u64 {
        set_zobrist_hash(&random_table, &signature.set_variables)
    }
}

pub fn create_set_zobrist_hash_with_others(
    model: &Model,
    zero_probability: Option<f64>,
) -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u64 = 42;

    let random_table = if let Some(p) = zero_probability {
        create_abstracted_random_table(model, SEED, p)
    } else {
        create_random_table(model, SEED)
    };

    move |signature: &HashableSignatureVariables| -> u64 {
        let mut hasher = FxHasher::default();
        let value = set_zobrist_hash(&random_table, &signature.set_variables);
        hasher.write_u64(value);

        signature.element_variables.hash(&mut hasher);
        signature.integer_variables.hash(&mut hasher);
        signature.continuous_variables.hash(&mut hasher);

        hasher.finish()
    }
}
