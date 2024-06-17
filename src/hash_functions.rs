use dypdl::prelude::*;
use dypdl_heuristic_search::search_algorithm::data_structure::HashableSignatureVariables;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use wyhash::WyHash;

use crate::state_serializer::StateSerializer;

pub fn create_fx_hash() -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u32 = 0x5583c24d;

    |signature: &HashableSignatureVariables| -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write_u32(SEED);
        signature.hash(&mut hasher);
        hasher.finish()
    }
}

pub fn create_wyhash() -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u64 = 0x5583c24d;

    |signature: &HashableSignatureVariables| -> u64 {
        let mut hasher = WyHash::with_seed(SEED);
        signature.hash(&mut hasher);
        hasher.finish()
    }
}

pub fn create_set_masks(model: &Model, zero_probability: f64) -> Vec<Vec<u32>> {
    const SEED: u64 = 42;

    let mut rng = Pcg64Mcg::seed_from_u64(SEED);
    let n = model.state_metadata.number_of_set_variables();
    let mut masks = Vec::with_capacity(n);

    for i in 0..n {
        let object_id = model.state_metadata.set_variable_to_object[i];
        let bits = model.state_metadata.object_numbers[object_id];
        let n_blocks = StateSerializer::compute_n_blocks(bits);
        let mut row = Vec::with_capacity(n_blocks);

        for _ in 0..n_blocks {
            let mut mask = 0;

            for _ in 0..32 {
                if rng.gen_bool(zero_probability) {
                    mask <<= 1;
                } else {
                    mask = (mask << 1) | 1;
                }
            }

            row.push(mask);
        }

        masks.push(row);
    }

    masks
}

pub fn create_masked_fx_hash(masks: Vec<Vec<u32>>) -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u32 = 0x5583c24d;

    move |signature: &HashableSignatureVariables| -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write_u32(SEED);

        for (v, row) in signature.set_variables.iter().zip(masks.iter()) {
            for (bits, mask) in v.as_slice().iter().zip(row.iter()) {
                hasher.write_u32(bits & mask);
            }
        }

        hasher.finish()
    }
}

pub fn create_masked_wyhash(masks: Vec<Vec<u32>>) -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u64 = 0x5583c24d;

    move |signature: &HashableSignatureVariables| -> u64 {
        let mut hasher = WyHash::with_seed(SEED);

        for (v, row) in signature.set_variables.iter().zip(masks.iter()) {
            for (bits, mask) in v.as_slice().iter().zip(row.iter()) {
                hasher.write_u32(bits & mask);
            }
        }

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

pub fn create_abstract_bytewise_random_table(
    model: &Model,
    zero_probability: f64,
) -> Vec<Vec<[[u64; 256]; 4]>> {
    const SEED: u64 = 42;

    let mut rng = Pcg64Mcg::seed_from_u64(SEED);
    let n = model.state_metadata.number_of_set_variables();
    let mut random_table = Vec::with_capacity(n);

    for i in 0..n {
        let object_id = model.state_metadata.set_variable_to_object[i];
        let bits = model.state_metadata.object_numbers[object_id];
        let n_blocks = StateSerializer::compute_n_blocks(bits);
        let mut vec_for_one_variable = Vec::with_capacity(n_blocks);

        for _ in 0..n_blocks {
            let mut bitwise_values = Vec::with_capacity(32);

            for _ in 0..32 {
                if rng.gen_bool(zero_probability) {
                    let rand = rng.gen::<u64>();
                    bitwise_values.push(vec![rand, rand]);
                } else {
                    bitwise_values.push(vec![rng.gen::<u64>(), rng.gen::<u64>()]);
                }
            }

            let mut array_for_one_block: [[u64; 256]; 4] = [[0; 256]; 4];

            for j in 0..4 {
                for k in 0..=255u8 {
                    let first_bit = (k & 0x1) as usize;
                    let second_bit = ((k >> 1) & 0x1) as usize;
                    let third_bit = ((k >> 2) & 0x1) as usize;
                    let fourth_bit = ((k >> 3) & 0x1) as usize;
                    let fifth_bit = ((k >> 4) & 0x1) as usize;
                    let sixth_bit = ((k >> 5) & 0x1) as usize;
                    let seventh_bit = ((k >> 6) & 0x1) as usize;
                    let eighth_bit = ((k >> 7) & 0x1) as usize;
                    array_for_one_block[j][k as usize] = bitwise_values[j * 8][first_bit]
                        ^ bitwise_values[j * 8 + 1][second_bit]
                        ^ bitwise_values[j * 8 + 2][third_bit]
                        ^ bitwise_values[j * 8 + 3][fourth_bit]
                        ^ bitwise_values[j * 8 + 4][fifth_bit]
                        ^ bitwise_values[j * 8 + 5][sixth_bit]
                        ^ bitwise_values[j * 8 + 6][seventh_bit]
                        ^ bitwise_values[j * 8 + 7][eighth_bit];
                }
            }

            vec_for_one_variable.push(array_for_one_block);
        }

        random_table.push(vec_for_one_variable);
    }

    random_table
}

fn bytewise_zobrist_hash(random_table: &[Vec<[[u64; 256]; 4]>], set_variables: &[Set]) -> u64 {
    let mut hash_value = 0;

    for (v, table) in set_variables.iter().zip(random_table.iter()) {
        for (bits, t) in v.as_slice().iter().zip(table.iter()) {
            let first_byte = bits & 0xff;
            let second_byte = (bits >> 8) & 0xff;
            let third_byte = (bits >> 16) & 0xff;
            let fourth_byte = (bits >> 24) & 0xff;
            hash_value ^= t[0][first_byte as usize]
                ^ t[1][second_byte as usize]
                ^ t[2][third_byte as usize]
                ^ t[3][fourth_byte as usize]
        }
    }

    hash_value
}

pub fn create_bytewise_zobrist_hash(
    random_table: Vec<Vec<[[u64; 256]; 4]>>,
) -> impl Fn(&HashableSignatureVariables) -> u64 {
    move |signature: &HashableSignatureVariables| -> u64 {
        bytewise_zobrist_hash(&random_table, &signature.set_variables)
    }
}

pub fn create_bytewise_zobrist_hash_with_others(
    random_table: Vec<Vec<[[u64; 256]; 4]>>,
) -> impl Fn(&HashableSignatureVariables) -> u64 {
    move |signature: &HashableSignatureVariables| -> u64 {
        let mut hasher = FxHasher::default();
        let value = bytewise_zobrist_hash(&random_table, &signature.set_variables);
        hasher.write_u64(value);

        signature.element_variables.hash(&mut hasher);
        signature.integer_variables.hash(&mut hasher);
        signature.continuous_variables.hash(&mut hasher);

        hasher.finish()
    }
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

fn popcount8(bits: u8) -> u8 {
    let mut n = bits;
    // use bit mask and bit operations
    n = (n & 0x55) + ((n >> 1) & 0x55);
    n = (n & 0x33) + ((n >> 2) & 0x33);
    (n & 0x0f) + ((n >> 4) & 0x0f)
}

fn create_4bits_field_random_table(
    model: &Model,
    seed: u64,
    abstraction_probability: f64,
) -> Vec<Vec<[[u64; 16]; 8]>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let n = model.state_metadata.number_of_set_variables();
    let mut random_table = Vec::with_capacity(n);

    for i in 0..n {
        let object_id = model.state_metadata.set_variable_to_object[i];
        let bits = model.state_metadata.object_numbers[object_id];
        let n_blocks = StateSerializer::compute_n_blocks(bits);
        let random_row = (0..n_blocks)
            .map(|_| {
                let mut randoms = [[0; 16]; 8];

                for r in &mut randoms {
                    if rng.gen_bool(abstraction_probability) {
                        let rand1 = rng.gen::<u64>();
                        let rand2 = rng.gen::<u64>();
                        let rand3 = rng.gen::<u64>();

                        for i in 0u8..16u8 {
                            let count = popcount8(i);

                            if count <= 1 {
                                r[i as usize] = rand1;
                            } else if count == 2 {
                                r[i as usize] = rand2;
                            } else {
                                r[i as usize] = rand3;
                            }
                        }
                    } else {
                        for rr in r {
                            *rr = rng.gen::<u64>();
                        }
                    }
                }

                randoms
            })
            .collect::<Vec<_>>();
        random_table.push(random_row);
    }

    random_table
}

fn compute_4bits_field_zobrist_hash(
    random_table: &[Vec<[[u64; 16]; 8]>],
    set_variables: &[Set],
) -> u64 {
    let mut hash_value = 0;

    for (v, table) in set_variables.iter().zip(random_table.iter()) {
        for (bits, t) in v.as_slice().iter().zip(table) {
            let first_4bits = bits & 0xf;
            let second_4bits = (bits >> 4) & 0xf;
            let third_4bits = (bits >> 8) & 0xf;
            let fourth_4bits = (bits >> 12) & 0xf;
            let fifth_4bits = (bits >> 16) & 0xf;
            let sixth_4bits = (bits >> 20) & 0xf;
            let seventh_4bits = (bits >> 24) & 0xf;
            let eighth_4bits = (bits >> 28) & 0xf;
            hash_value ^= t[0][first_4bits as usize]
                ^ t[1][second_4bits as usize]
                ^ t[2][third_4bits as usize]
                ^ t[3][fourth_4bits as usize]
                ^ t[4][fifth_4bits as usize]
                ^ t[5][sixth_4bits as usize]
                ^ t[6][seventh_4bits as usize]
                ^ t[7][eighth_4bits as usize];
        }
    }

    hash_value
}

pub fn create_4bits_field_zobrist_hash(
    model: &Model,
    abstraction_probability: f64,
) -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u64 = 42;

    let random_table = create_4bits_field_random_table(model, SEED, abstraction_probability);

    move |signature: &HashableSignatureVariables| -> u64 {
        compute_4bits_field_zobrist_hash(&random_table, &signature.set_variables)
    }
}

struct ThreeBitsFieldRandomTable {
    randoms: [[u64; 8]; 10],
    last_randoms: [u64; 4],
}

fn create_3bits_field_random_table(
    model: &Model,
    seed: u64,
    abstraction_probability: f64,
) -> Vec<Vec<ThreeBitsFieldRandomTable>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let n = model.state_metadata.number_of_set_variables();
    let mut random_table = Vec::with_capacity(n);

    for i in 0..n {
        let object_id = model.state_metadata.set_variable_to_object[i];
        let bits = model.state_metadata.object_numbers[object_id];
        let n_blocks = StateSerializer::compute_n_blocks(bits);
        let random_row = (0..n_blocks)
            .map(|_| {
                let mut randoms = [[0; 8]; 10];

                for r in &mut randoms {
                    if rng.gen_bool(abstraction_probability) {
                        let rand1 = rng.gen::<u64>();
                        let rand2 = rng.gen::<u64>();

                        for i in 0u8..8u8 {
                            if i == 0 || i == 1 || i == 2 || i == 4 {
                                r[i as usize] = rand1;
                            } else {
                                r[i as usize] = rand2;
                            }
                        }
                    } else {
                        for rr in r {
                            *rr = rng.gen::<u64>();
                        }
                    }
                }

                let last_randoms = [
                    rng.gen::<u64>(),
                    rng.gen::<u64>(),
                    rng.gen::<u64>(),
                    rng.gen::<u64>(),
                ];

                ThreeBitsFieldRandomTable {
                    randoms,
                    last_randoms,
                }
            })
            .collect::<Vec<_>>();
        random_table.push(random_row);
    }

    random_table
}

fn compute_3bits_field_zobrist_hash(
    random_table: &[Vec<ThreeBitsFieldRandomTable>],
    set_variables: &[Set],
) -> u64 {
    let mut hash_value = 0;

    for (v, table) in set_variables.iter().zip(random_table.iter()) {
        for (bits, t) in v.as_slice().iter().zip(table) {
            let first_3bits = bits & 0x7;
            let second_3bits = (bits >> 3) & 0x7;
            let third_3bits = (bits >> 6) & 0x7;
            let fourth_3bits = (bits >> 9) & 0x7;
            let fifth_3bits = (bits >> 12) & 0x7;
            let sixth_3bits = (bits >> 15) & 0x7;
            let seventh_3bits = (bits >> 18) & 0x7;
            let eighth_3bits = (bits >> 21) & 0x7;
            let ninth_3bits = (bits >> 24) & 0x7;
            let tenth_3bits = (bits >> 27) & 0x7;
            let last_2bits = bits >> 30;
            hash_value ^= t.randoms[0][first_3bits as usize]
                ^ t.randoms[1][second_3bits as usize]
                ^ t.randoms[2][third_3bits as usize]
                ^ t.randoms[3][fourth_3bits as usize]
                ^ t.randoms[4][fifth_3bits as usize]
                ^ t.randoms[5][sixth_3bits as usize]
                ^ t.randoms[6][seventh_3bits as usize]
                ^ t.randoms[7][eighth_3bits as usize]
                ^ t.randoms[8][ninth_3bits as usize]
                ^ t.randoms[9][tenth_3bits as usize]
                ^ t.last_randoms[last_2bits as usize];
        }
    }

    hash_value
}

pub fn create_3bits_field_zobrist_hash(
    model: &Model,
    abstraction_probability: f64,
) -> impl Fn(&HashableSignatureVariables) -> u64 {
    const SEED: u64 = 42;

    let random_table = create_3bits_field_random_table(model, SEED, abstraction_probability);

    move |signature: &HashableSignatureVariables| -> u64 {
        compute_3bits_field_zobrist_hash(&random_table, &signature.set_variables)
    }
}
