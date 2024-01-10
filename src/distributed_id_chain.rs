use mpi::traits::*;
use mpi::{datatype::DatatypeRef, Address, Count};
use std::cell::Cell;
use std::mem::size_of;
use std::rc::Rc;
use zerocopy::{AsBytes, FromBytes};

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct DistributedTransitionIdChainData {
    parent_chain_id: usize,
    last_transition_id: usize,
    last_forced: bool,
}

#[derive(Default, PartialEq, Eq, Clone, Debug)]
pub struct DistributedTransitionIdChain {
    pub id: Cell<Option<usize>>,
    pub parent_rank: Option<i32>,
    data: Option<DistributedTransitionIdChainData>,
}

impl DistributedTransitionIdChain {
    pub fn generate_successor(&self, transition_id: usize, forced: bool) -> Self {
        let data = DistributedTransitionIdChainData {
            parent_chain_id: self.id.get().unwrap(),
            last_transition_id: transition_id,
            last_forced: forced,
        };

        DistributedTransitionIdChain {
            id: Cell::new(None),
            parent_rank: None,
            data: Some(data),
        }
    }

    pub fn get_transition_ids_in_this_rank(
        &self,
        id_to_chain_node: &[Rc<Self>],
    ) -> (Vec<(usize, bool)>, Option<i32>) {
        let mut result = Vec::default();
        let mut chain = self;

        while let Some(data) = chain.data.as_ref() {
            result.push((data.last_transition_id, data.last_forced));
            chain = &id_to_chain_node[data.parent_chain_id];
        }

        result.reverse();

        (result, chain.parent_rank)
    }

    pub fn get_total_size() -> usize {
        size_of::<i32>() + 2 * size_of::<usize>() + size_of::<u8>()
    }

    pub fn get_datatype_blocklengths() -> [Count; 3] {
        [1, 2, 1]
    }

    pub fn get_datatype_displacements() -> [Address; 3] {
        [
            0 as Address,
            size_of::<i32>() as Address,
            (size_of::<i32>() + 2 * size_of::<usize>()) as Address,
        ]
    }

    pub fn get_datatype_types() -> [DatatypeRef<'static>; 3] {
        [
            i32::equivalent_datatype(),
            usize::equivalent_datatype(),
            u8::equivalent_datatype(),
        ]
    }

    pub fn serialize_to(&self, buffer: &mut [u8]) {
        let mut offset = 0;

        let parent_rank = self.parent_rank.unwrap();
        let data = self.data.as_ref().unwrap();

        let bytes = parent_rank.as_bytes();
        let size = bytes.len();
        buffer[offset..offset + size].copy_from_slice(bytes);
        offset += size;

        let bytes = data.parent_chain_id.as_bytes();
        let size = bytes.len();
        buffer[offset..offset + size].copy_from_slice(bytes);
        offset += size;

        let bytes = data.last_transition_id.as_bytes();
        let size = bytes.len();
        buffer[offset..offset + size].copy_from_slice(bytes);
        offset += size;

        let last_forced = if data.last_forced { 1u8 } else { 0u8 };
        let bytes = last_forced.as_bytes();
        let size = bytes.len();
        buffer[offset..offset + size].copy_from_slice(bytes);
    }

    pub fn deserialize(buffer: &[u8]) -> Self {
        let mut offset = 0;

        let size = size_of::<i32>();
        let parent_rank = i32::read_from(&buffer[offset..size]).unwrap();
        offset += size;

        let size = size_of::<usize>();
        let parent_chain_id = usize::read_from(&buffer[offset..size]).unwrap();
        offset += size;

        let size = size_of::<usize>();
        let last_transition_id = usize::read_from(&buffer[offset..size]).unwrap();
        offset += size;

        let size = size_of::<u8>();
        let last_forced = u8::read_from(&buffer[offset..size]).unwrap();
        let last_forced = last_forced == 1u8;

        DistributedTransitionIdChain {
            id: Cell::new(None),
            parent_rank: Some(parent_rank),
            data: Some(DistributedTransitionIdChainData {
                parent_chain_id,
                last_transition_id,
                last_forced,
            }),
        }
    }
}

pub trait GetDistributedTransitionIdChain {
    fn get_distributed_transition_id_chain(&self) -> &Rc<DistributedTransitionIdChain>;
}
