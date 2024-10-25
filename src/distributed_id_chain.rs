use mpi::{datatype::DatatypeRef, traits::*, Address, Count, Rank};
use std::cell::Cell;
use std::mem;
use std::rc::Rc;
use zerocopy::{FromBytes, IntoBytes};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedTransitionIdChainData {
    parent_chain_id: usize,
    last_transition_id: usize,
    last_transition_forced: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DistributedTransitionIdChain {
    pub id: Cell<Option<usize>>,
    pub parent_rank: Cell<Option<Rank>>,
    data: Option<DistributedTransitionIdChainData>,
}

impl DistributedTransitionIdChain {
    pub fn get_parent_id(&self) -> Option<usize> {
        self.data.as_ref().map(|data| data.parent_chain_id)
    }

    pub fn generate_successor(&self, transition_id: usize, forced: bool) -> Self {
        let data = DistributedTransitionIdChainData {
            parent_chain_id: self.id.get().unwrap(),
            last_transition_id: transition_id,
            last_transition_forced: forced,
        };

        DistributedTransitionIdChain {
            id: Cell::new(None),
            parent_rank: Cell::new(None),
            data: Some(data),
        }
    }

    pub fn get_transition_ids_in_this_rank(
        &self,
        id_to_chain_node: &[Rc<Self>],
    ) -> (Vec<usize>, Vec<bool>, Option<(Rank, usize)>) {
        let mut ids = Vec::default();
        let mut forced = Vec::default();
        let mut chain = self;

        while let Some(data) = chain.data.as_ref() {
            ids.push(data.last_transition_id);
            forced.push(data.last_transition_forced);

            if chain.parent_rank.get().is_some() {
                break;
            }

            chain = &id_to_chain_node[data.parent_chain_id];
        }

        (
            ids,
            forced,
            chain
                .parent_rank
                .get()
                .map(|rank| (rank, chain.data.as_ref().unwrap().parent_chain_id)),
        )
    }

    pub fn get_total_size() -> usize {
        mem::size_of::<Rank>() + 2 * mem::size_of::<usize>() + mem::size_of::<u8>()
    }

    pub fn get_datatype_blocklengths() -> [Count; 3] {
        [1, 2, 1]
    }

    pub fn get_datatype_displacements() -> [Address; 3] {
        [
            0 as Address,
            mem::size_of::<Rank>() as Address,
            (mem::size_of::<Rank>() + 2 * mem::size_of::<usize>()) as Address,
        ]
    }

    pub fn get_datatype_types() -> [DatatypeRef<'static>; 3] {
        [
            Rank::equivalent_datatype(),
            usize::equivalent_datatype(),
            u8::equivalent_datatype(),
        ]
    }

    pub fn serialize_to(&self, buffer: &mut [u8]) {
        let mut offset = 0;

        let parent_rank = self.parent_rank.get().unwrap();
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

        let last_forced = if data.last_transition_forced {
            1u8
        } else {
            0u8
        };
        let bytes = last_forced.as_bytes();
        let size = bytes.len();
        buffer[offset..offset + size].copy_from_slice(bytes);
    }

    pub fn deserialize(buffer: &[u8]) -> Self {
        let mut offset = 0;

        let size = mem::size_of::<Rank>();
        let parent_rank = Rank::read_from_bytes(&buffer[offset..offset + size]).unwrap();
        offset += size;

        let size = mem::size_of::<usize>();
        let parent_chain_id = usize::read_from_bytes(&buffer[offset..offset + size]).unwrap();
        offset += size;

        let size = mem::size_of::<usize>();
        let last_transition_id = usize::read_from_bytes(&buffer[offset..offset + size]).unwrap();
        offset += size;

        let size = mem::size_of::<u8>();
        let last_forced = u8::read_from_bytes(&buffer[offset..offset + size]).unwrap();
        let last_forced = last_forced == 1u8;

        DistributedTransitionIdChain {
            id: Cell::new(None),
            parent_rank: Cell::new(Some(parent_rank)),
            data: Some(DistributedTransitionIdChainData {
                parent_chain_id,
                last_transition_id,
                last_transition_forced: last_forced,
            }),
        }
    }
}

pub trait GetDistributedTransitionIdChain {
    fn get_distributed_transition_id_chain(&self) -> &DistributedTransitionIdChain;
}

pub trait GeRcDistributedTransitionIdChain: GetDistributedTransitionIdChain {
    fn get_rc_distributed_transition_id_chain(&self) -> &Rc<DistributedTransitionIdChain>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let chain = DistributedTransitionIdChain::default();

        assert_eq!(chain.id.get(), None);
        assert_eq!(chain.parent_rank.get(), None);
        assert_eq!(chain.data, None);
    }

    #[test]
    fn test_get_parent_id_some() {
        let chain = DistributedTransitionIdChain::default();
        chain.id.set(Some(0));
        let successor = chain.generate_successor(0, false);

        assert_eq!(successor.get_parent_id(), Some(0));
    }

    #[test]
    fn test_get_parent_id_none() {
        let chain = DistributedTransitionIdChain::default();
        assert_eq!(chain.get_parent_id(), None);
    }

    #[test]
    fn test_generate_successor() {
        let chain = DistributedTransitionIdChain::default();
        chain.id.set(Some(0));
        let successor = chain.generate_successor(0, false);

        assert_eq!(successor.id.get(), None);
        assert_eq!(successor.parent_rank.get(), None);
        assert_eq!(
            successor.data,
            Some(DistributedTransitionIdChainData {
                parent_chain_id: 0,
                last_transition_id: 0,
                last_transition_forced: false
            })
        );
    }

    #[test]
    fn test_get_transition_ids_in_this_rank() {
        let chain = Rc::new(DistributedTransitionIdChain::default());
        chain.id.set(Some(0));

        let mut id_to_chain_node = vec![];

        let successor = Rc::new(chain.generate_successor(0, false));
        successor.parent_rank.set(Some(0));
        successor.id.set(Some(0));
        id_to_chain_node.push(successor.clone());

        let successor = Rc::new(successor.generate_successor(1, true));
        successor.id.set(Some(1));
        id_to_chain_node.push(successor.clone());

        let (transition_ids, transition_forced, parent) =
            successor.get_transition_ids_in_this_rank(&id_to_chain_node);

        assert_eq!(transition_ids, vec![1, 0]);
        assert_eq!(transition_forced, vec![true, false]);
        assert_eq!(parent, Some((0, 0)));
    }

    #[test]
    fn test_serialize_deserialize() {
        let chain = DistributedTransitionIdChain::default();
        chain.id.set(Some(0));
        let successor = chain.generate_successor(0, false);
        successor.parent_rank.set(Some(1));

        let mut buffer = vec![0u8; DistributedTransitionIdChain::get_total_size()];
        successor.serialize_to(&mut buffer);

        let deserialized = DistributedTransitionIdChain::deserialize(&buffer);

        assert_eq!(deserialized, successor);
    }
}
