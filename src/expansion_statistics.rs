use mpi::{traits::*, Rank, Tag};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpansionStatistics<T> {
    counts: BTreeMap<(usize, Option<T>, T), usize>,
}

impl<T> Default for ExpansionStatistics<T> {
    fn default() -> Self {
        Self {
            counts: BTreeMap::new(),
        }
    }
}

impl<T: Ord> ExpansionStatistics<T> {
    pub fn record(&mut self, depth: usize, f_value: Option<T>, g_value: T) {
        *self.counts.entry((depth, f_value, g_value)).or_default() += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, Option<&T>, &T, usize)> {
        self.counts
            .iter()
            .map(|((depth, f_value, g_value), count)| (*depth, f_value.as_ref(), g_value, *count))
    }

    pub fn total(&self) -> usize {
        self.counts.values().sum()
    }
}

impl<T: Ord + Display> ExpansionStatistics<T> {
    pub fn dump_to_csv(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filename)?;
        file.write_all(b"depth,f_value,g_value,expanded\n")?;

        for (depth, f_value, g_value, count) in self.iter() {
            let f_value = f_value.map_or_else(String::new, ToString::to_string);
            writeln!(file, "{depth},{f_value},{g_value},{count}")?;
        }

        Ok(())
    }

    fn serialize(&self) -> String {
        let mut serialized = String::new();
        for (depth, f_value, g_value, count) in self.iter() {
            let f_value = f_value.map_or_else(String::new, ToString::to_string);
            serialized.push_str(&format!("{depth},{f_value},{g_value},{count}\n"));
        }
        serialized
    }
}

impl<T> ExpansionStatistics<T>
where
    T: Ord + Display + FromStr,
    <T as FromStr>::Err: Debug,
{
    fn merge_serialized(&mut self, serialized: &[u8]) -> Result<(), Box<dyn Error>> {
        let serialized = std::str::from_utf8(serialized)?;
        for line in serialized.lines() {
            let mut fields = line.splitn(4, ',');
            let depth = fields
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing depth"))?
                .parse::<usize>()?;
            let f_value = fields
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing f-value"))?;
            let f_value = if f_value.is_empty() {
                None
            } else {
                Some(f_value.parse::<T>().map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid f-value: {error:?}"),
                    )
                })?)
            };
            let g_value = fields
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing g-value"))?
                .parse::<T>()
                .map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid g-value: {error:?}"),
                    )
                })?;
            let count = fields
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing count"))?
                .parse::<usize>()?;
            *self.counts.entry((depth, f_value, g_value)).or_default() += count;
        }
        Ok(())
    }

    /// Gathers and aggregates statistics at `root_rank`.
    pub fn gather<C: Communicator>(
        &self,
        communicator: &C,
        root_rank: Rank,
        tag: Tag,
    ) -> Result<Option<Self>, Box<dyn Error>> {
        let serialized = self.serialize();
        if communicator.rank() == root_rank {
            let mut aggregated = Self::default();
            aggregated.merge_serialized(serialized.as_bytes())?;
            for rank in 0..communicator.size() {
                if rank == root_rank {
                    continue;
                }
                let (serialized, _) = communicator
                    .process_at_rank(rank)
                    .receive_vec_with_tag::<u8>(tag);
                aggregated.merge_serialized(&serialized)?;
            }
            Ok(Some(aggregated))
        } else {
            communicator
                .process_at_rank(root_rank)
                .send_with_tag(serialized.as_bytes(), tag);
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_by_depth_f_value_and_g_value() {
        let mut statistics = ExpansionStatistics::default();
        statistics.record(1, Some(3), 1);
        statistics.record(1, Some(3), 1);
        statistics.record(1, Some(3), 2);
        statistics.record(2, Some(3), 2);
        statistics.record(2, None, 2);

        assert_eq!(
            statistics.iter().collect::<Vec<_>>(),
            vec![
                (1, Some(&3), &1, 2),
                (1, Some(&3), &2, 1),
                (2, None, &2, 1),
                (2, Some(&3), &2, 1),
            ]
        );
        assert_eq!(statistics.total(), 5);
    }

    #[test]
    fn serialize_and_merge() {
        let mut first = ExpansionStatistics::default();
        first.record(1, Some(3), 1);
        first.record(2, None, 2);
        let mut second = ExpansionStatistics::default();
        second.record(1, Some(3), 1);
        second.record(1, Some(3), 2);
        second.record(2, Some(4), 2);

        first
            .merge_serialized(second.serialize().as_bytes())
            .unwrap();

        assert_eq!(
            first.iter().collect::<Vec<_>>(),
            vec![
                (1, Some(&3), &1, 2),
                (1, Some(&3), &2, 1),
                (2, None, &2, 1),
                (2, Some(&4), &2, 1),
            ]
        );
        assert_eq!(first.total(), 5);
    }
}
