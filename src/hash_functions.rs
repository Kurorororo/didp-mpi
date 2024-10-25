use dypdl::prelude::*;
use dypdl_heuristic_search::search_algorithm::data_structure::HashableSignatureVariables;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

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
