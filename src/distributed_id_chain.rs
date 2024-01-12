use memoffset::offset_of;
use mpi::datatype::UserDatatype;
use mpi::traits::*;
use mpi::{datatype::DatatypeRef, Address, Count, Rank};
use std::cell::Cell;
use std::mem::size_of;
use std::rc::Rc;
use zerocopy::{AsBytes, FromBytes};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransitionId(pub usize, pub bool);

unsafe impl Equivalence for TransitionId {
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(TransitionId, 0) as Address,
                offset_of!(TransitionId, 1) as Address,
            ],
            &[usize::equivalent_datatype(), bool::equivalent_datatype()],
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedTransitionIdChainData {
    parent_chain_id: usize,
    last_transition_id: TransitionId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DistributedTransitionIdChain {
    pub id: Cell<Option<usize>>,
    pub parent_rank: Cell<Option<Rank>>,
    data: Option<DistributedTransitionIdChainData>,
}

impl DistributedTransitionIdChain {
    pub fn generate_successor(&self, transition_id: TransitionId) -> Self {
        let data = DistributedTransitionIdChainData {
            parent_chain_id: self.id.get().unwrap(),
            last_transition_id: transition_id,
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
    ) -> (Vec<TransitionId>, Option<(Rank, usize)>) {
        let mut ids = Vec::default();
        let mut chain = self;

        while let Some(data) = chain.data.as_ref() {
            ids.push(data.last_transition_id);

            if chain.parent_rank.get().is_some() {
                break;
            }

            chain = &id_to_chain_node[data.parent_chain_id];
        }

        (
            ids,
            chain
                .parent_rank
                .get()
                .map(|rank| (rank, chain.data.as_ref().unwrap().parent_chain_id)),
        )
    }

    pub fn get_total_size() -> usize {
        size_of::<Rank>() + 2 * size_of::<usize>() + size_of::<u8>()
    }

    pub fn get_datatype_blocklengths() -> [Count; 3] {
        [1, 2, 1]
    }

    pub fn get_datatype_displacements() -> [Address; 3] {
        [
            0 as Address,
            size_of::<Rank>() as Address,
            (size_of::<Rank>() + 2 * size_of::<usize>()) as Address,
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

        let bytes = data.last_transition_id.0.as_bytes();
        let size = bytes.len();
        buffer[offset..offset + size].copy_from_slice(bytes);
        offset += size;

        let last_forced = if data.last_transition_id.1 { 1u8 } else { 0u8 };
        let bytes = last_forced.as_bytes();
        let size = bytes.len();
        buffer[offset..offset + size].copy_from_slice(bytes);
    }

    pub fn deserialize(buffer: &[u8]) -> Self {
        let mut offset = 0;

        let size = size_of::<Rank>();
        let parent_rank = Rank::read_from(&buffer[offset..offset + size]).unwrap();
        offset += size;

        let size = size_of::<usize>();
        let parent_chain_id = usize::read_from(&buffer[offset..offset + size]).unwrap();
        offset += size;

        let size = size_of::<usize>();
        let last_transition_id = usize::read_from(&buffer[offset..offset + size]).unwrap();
        offset += size;

        let size = size_of::<u8>();
        let last_forced = u8::read_from(&buffer[offset..offset + size]).unwrap();
        let last_forced = last_forced == 1u8;

        DistributedTransitionIdChain {
            id: Cell::new(None),
            parent_rank: Cell::new(Some(parent_rank)),
            data: Some(DistributedTransitionIdChainData {
                parent_chain_id,
                last_transition_id: TransitionId(last_transition_id, last_forced),
            }),
        }
    }
}

pub trait GetDistributedTransitionIdChain {
    fn get_distributed_transition_id_chain(&self) -> &Rc<DistributedTransitionIdChain>;
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
    fn test_generate_successor() {
        let chain = DistributedTransitionIdChain::default();
        chain.id.set(Some(0));
        let successor = chain.generate_successor(TransitionId(0, false));

        assert_eq!(successor.id.get(), None);
        assert_eq!(successor.parent_rank.get(), None);
        assert_eq!(
            successor.data,
            Some(DistributedTransitionIdChainData {
                parent_chain_id: 0,
                last_transition_id: TransitionId(0, false),
            })
        );
    }

    #[test]
    fn test_get_transition_ids_in_this_rank() {
        let chain = Rc::new(DistributedTransitionIdChain::default());
        chain.id.set(Some(0));

        let mut id_to_chain_node = vec![];

        let successor = Rc::new(chain.generate_successor(TransitionId(0, false)));
        successor.parent_rank.set(Some(0));
        successor.id.set(Some(0));
        id_to_chain_node.push(successor.clone());

        let successor = Rc::new(successor.generate_successor(TransitionId(1, true)));
        successor.id.set(Some(1));
        id_to_chain_node.push(successor.clone());

        let (transition_ids, parent) = successor.get_transition_ids_in_this_rank(&id_to_chain_node);

        assert_eq!(
            transition_ids,
            vec![TransitionId(1, true), TransitionId(0, false)]
        );
        assert_eq!(parent, Some((0, 0)));
    }

    #[test]
    fn test_serialize_deserialize() {
        let chain = DistributedTransitionIdChain::default();
        chain.id.set(Some(0));
        let successor = chain.generate_successor(TransitionId(0, false));
        successor.parent_rank.set(Some(1));

        let mut buffer = vec![0u8; DistributedTransitionIdChain::get_total_size()];
        successor.serialize_to(&mut buffer);

        let deserialized = DistributedTransitionIdChain::deserialize(&buffer);

        assert_eq!(deserialized, successor);
    }
}
