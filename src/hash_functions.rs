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

fn create_per_element_random_table(model: &Model, seed: u64) -> Vec<Vec<Vec<u64>>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let n = model.state_metadata.number_of_set_variables();
    let mut random_table = Vec::with_capacity(n);

    for i in 0..n {
        let object_id = model.state_metadata.set_variable_to_object[i];
        let m = model.state_metadata.object_numbers[object_id];
        let random_row = (0..m)
            .map(|_| vec![rng.gen::<u64>(), rng.gen::<u64>()])
            .collect::<Vec<_>>();
        random_table.push(random_row);
    }

    random_table
}

fn create_abstracted_per_element_random_table(
    model: &Model,
    seed: u64,
    zero_probability: f64,
) -> Vec<Vec<Vec<u64>>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let n = model.state_metadata.number_of_set_variables();
    let mut random_table = Vec::with_capacity(n);

    for i in 0..n {
        let object_id = model.state_metadata.set_variable_to_object[i];
        let m = model.state_metadata.object_numbers[object_id];
        let random_row = (0..m)
            .map(|_| {
                if rng.gen_bool(zero_probability) {
                    let rand = rng.gen::<u64>();
                    vec![rand, rand]
                } else {
                    vec![rng.gen::<u64>(), rng.gen::<u64>()]
                }
            })
            .collect::<Vec<_>>();
        random_table.push(random_row);
    }

    random_table
}

// fn precompute_consecutive_xor(random_table: &[Vec<Vec<u64>>]) -> Vec<Vec<Vec<Vec<u64>>>> {
//     let mut result = Vec::with_capacity(random_table.len());
//
//     for table in random_table {
//         let n = table.len();
//         let mut result_table = Vec::with_capacity(n);
//
//         // XOR of j+1 consecutive random numbers starting from index i.
//         for i in 0..n {
//             let mut row = Vec::with_capacity(n);
//
//             for j in 1..=n - i {
//                 let r0 = table
//                     .iter()
//                     .map(|x| x[0])
//                     .skip(i)
//                     .take(j)
//                     .reduce(|a, b| a ^ b)
//                     .unwrap();
//                 let r1 = table
//                     .iter()
//                     .map(|x| x[1])
//                     .skip(i)
//                     .take(j)
//                     .reduce(|a, b| a ^ b)
//                     .unwrap();
//                 row.push(vec![r0, r1]);
//             }
//
//             result_table.push(row);
//         }
//
//         result.push(result_table);
//     }
//
//     result
// }

// fn set_per_element_zobrist_hash(random_table: &[Vec<Vec<Vec<u64>>>], set_variables: &[Set]) -> u64 {
//     let mut hash_value = 0;
//
//     for (v, table) in set_variables.iter().zip(random_table.iter()) {
//         let mut previous_one = None;
//         let mut consecutive_ones = 1;
//
//         for i in v.ones() {
//             if let Some(previous_one) = previous_one {
//                 if i == previous_one + 1 {
//                     consecutive_ones += 1;
//                 } else {
//                     let start_one: usize = previous_one - consecutive_ones + 1;
//                     hash_value ^= table[start_one][consecutive_ones - 1][1];
//                     let start_zero: usize = previous_one + 1;
//                     let consecutive_zeros: usize = i - previous_one - 1;
//                     hash_value ^= table[start_zero][consecutive_zeros - 1][0];
//                     consecutive_ones = 1;
//                 }
//             } else if i > 0 {
//                 hash_value ^= table[0][i - 1][0];
//             }
//
//             previous_one = Some(i);
//         }
//
//         if let Some(previous_one) = previous_one {
//             if previous_one < v.len() - 1 {
//                 let start_zero: usize = previous_one + 1;
//                 let consecutive_zeros: usize = v.len() - previous_one - 1;
//                 hash_value ^= table[start_zero][consecutive_zeros - 1][0];
//             }
//         } else {
//             hash_value ^= table[0][v.len() - 1][0];
//         }
//     }
//
//     hash_value
// }

fn set_per_element_zobrist_hash_naive(
    random_table: &[Vec<Vec<u64>>],
    set_variables: &[Set],
) -> u64 {
    let mut hash_value = 0;

    for (v, table) in set_variables.iter().zip(random_table.iter()) {
        let mut previous_one = None;

        for i in v.ones() {
            let previous_zero = previous_one.map_or(0, |x| x + 1);

            for r in &table[previous_zero..i] {
                hash_value ^= r[0];
            }

            hash_value ^= table[i][1];
            previous_one = Some(i);
        }

        let previous_zero = previous_one.map_or(0, |x| x + 1);

        for r in &table[previous_zero..] {
            hash_value ^= r[0];
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
        create_abstracted_per_element_random_table(model, SEED, p)
    } else {
        create_per_element_random_table(model, SEED)
    };
    //let random_table = precompute_consecutive_xor(&random_table);

    move |signature: &HashableSignatureVariables| -> u64 {
        //set_per_element_zobrist_hash(&random_table, &signature.set_variables)
        set_per_element_zobrist_hash_naive(&random_table, &signature.set_variables)
    }
}

pub fn create_set_zobrist_hash_with_others(
    model: &Model,
    zero_probability: Option<f64>,
) -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u64 = 42;

    let random_table = if let Some(p) = zero_probability {
        create_abstracted_per_element_random_table(model, SEED, p)
    } else {
        create_per_element_random_table(model, SEED)
    };
    //let random_table = precompute_consecutive_xor(&random_table);

    move |signature: &HashableSignatureVariables| -> u64 {
        let mut hasher = FxHasher::default();
        //let value = set_per_element_zobrist_hash(&random_table, &signature.set_variables);
        let value = set_per_element_zobrist_hash_naive(&random_table, &signature.set_variables);
        hasher.write_u64(value);

        signature.element_variables.hash(&mut hasher);
        signature.integer_variables.hash(&mut hasher);
        signature.continuous_variables.hash(&mut hasher);

        hasher.finish()
    }
}
