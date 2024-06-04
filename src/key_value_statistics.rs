use crate::is_float::IsFloat;
use dypdl::variable_type::Numeric;
use linked_hash_map::LinkedHashMap;
use mpi::{traits::*, Rank, Tag};
use std::collections::{BTreeMap, HashMap};
use yaml_rust::Yaml;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyValueStatistics<K, V> {
    pub n_keys: usize,
    pub keys: Vec<K>,
    pub values: Vec<V>,
}

impl<K: Ord + Clone, V: Clone, H> From<HashMap<K, V, H>> for KeyValueStatistics<K, V> {
    fn from(map: HashMap<K, V, H>) -> Self {
        let n_keys = map.len();
        let (keys, values): (Vec<K>, Vec<V>) = map.into_iter().unzip();
        let mut indices = (0..n_keys).collect::<Vec<usize>>();
        indices.sort_by_key(|&i| &keys[i]);
        let keys = indices.iter().map(|&i| keys[i].clone()).collect();
        let values = indices.iter().map(|&i| values[i].clone()).collect();

        Self {
            n_keys,
            keys,
            values,
        }
    }
}

impl<K, V, U: Ord + From<K>> From<KeyValueStatistics<K, V>> for BTreeMap<U, V> {
    fn from(value: KeyValueStatistics<K, V>) -> Self {
        value
            .keys
            .into_iter()
            .zip(value.values)
            .map(|(k, v)| (k.into(), v))
            .collect()
    }
}

impl<K: IsFloat> From<KeyValueStatistics<K, usize>> for Yaml {
    fn from(value: KeyValueStatistics<K, usize>) -> Self {
        let mut array = Vec::with_capacity(value.n_keys);

        for (k, v) in value.keys.iter().zip(value.values) {
            let k = if K::is_float() {
                Yaml::Real(k.to_continuous().to_string())
            } else {
                Yaml::Integer(k.to_integer() as i64)
            };
            let mut map = LinkedHashMap::default();
            map.insert(Yaml::String(String::from("key")), k);
            map.insert(Yaml::String(String::from("value")), Yaml::Integer(v as i64));
            array.push(Yaml::Hash(map));
        }

        Yaml::Array(array)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyValueStatisticsTags {
    pub n_keys_tag: Tag,
    pub keys_tag: Tag,
    pub values_tag: Tag,
}

impl<K, V> KeyValueStatistics<K, V>
where
    K: Ord + Numeric + IsFloat,
    V: Clone + Default,
    [V]: Buffer + BufferMut,
{
    pub fn send<C: Communicator>(
        &self,
        communicator: &C,
        destination_rank: Rank,
        tags: KeyValueStatisticsTags,
    ) {
        let destination = communicator.process_at_rank(destination_rank);

        if K::is_float() {
            let keys = self
                .keys
                .iter()
                .map(|k| k.to_continuous())
                .collect::<Vec<_>>();
            destination.send_with_tag(&keys[..], tags.keys_tag);
        } else {
            let keys = self.keys.iter().map(|k| k.to_integer()).collect::<Vec<_>>();
            destination.send_with_tag(&keys[..], tags.keys_tag);
        }

        destination.send_with_tag(&self.n_keys, tags.n_keys_tag);
        destination.send_with_tag(&self.values[..], tags.values_tag);
    }

    pub fn receive<C: Communicator>(
        communicator: &C,
        source_rank: Rank,
        tags: KeyValueStatisticsTags,
    ) -> Self {
        let source = communicator.process_at_rank(source_rank);
        let (n_keys, _) = source.receive_with_tag::<usize>(tags.n_keys_tag);

        let keys = if K::is_float() {
            let mut keys = vec![0.0; n_keys];
            source.receive_into_with_tag(&mut keys[..], tags.keys_tag);
            keys.into_iter().map(K::from).collect()
        } else {
            let mut keys = vec![0; n_keys];
            source.receive_into_with_tag(&mut keys[..], tags.keys_tag);
            keys.into_iter().map(K::from).collect()
        };

        let mut values = vec![V::default(); n_keys];
        source.receive_into_with_tag(&mut values[..], tags.values_tag);

        Self {
            n_keys,
            keys,
            values,
        }
    }

    pub fn gather<C: Communicator>(
        &self,
        communicator: &C,
        root_rank: Rank,
        tags: KeyValueStatisticsTags,
    ) -> Vec<Self> {
        if root_rank == communicator.rank() {
            let mut result = Vec::with_capacity(communicator.size() as usize);

            for rank in 0..communicator.size() {
                if rank == root_rank {
                    result.push(self.clone());
                } else {
                    let received = Self::receive(communicator, rank, tags);
                    result.push(received);
                }
            }

            result
        } else {
            self.send(communicator, root_rank, tags);
            Vec::default()
        }
    }
}
