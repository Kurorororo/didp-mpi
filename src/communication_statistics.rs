//! Exact send-side point-to-point volume; independent of clock sampling.
use mpi::{datatype::Buffer, traits::*, Rank};
use std::cell::RefCell;

#[derive(Clone, Copy)]
pub enum SendKind {
    Buffered = 0,
    Standard = 1,
}

/// Intra messages/bytes, inter messages/bytes, for Bsend then Send.
#[derive(Clone, Default)]
pub struct Report {
    pub counts: [[u64; 4]; 2],
    pub node_leader: Rank,
    pub ranks_on_node: Rank,
}

impl Report {
    pub const PACKED_LEN: usize = 10;

    pub fn pack(&self) -> [u64; Self::PACKED_LEN] {
        let mut result = [0; Self::PACKED_LEN];
        result[..4].copy_from_slice(&self.counts[0]);
        result[4..8].copy_from_slice(&self.counts[1]);
        result[8] = self.node_leader as u64;
        result[9] = self.ranks_on_node as u64;
        result
    }

    pub fn unpack(values: &[u64]) -> Self {
        assert_eq!(values.len(), Self::PACKED_LEN);
        Self {
            counts: [
                values[..4].try_into().unwrap(),
                values[4..8].try_into().unwrap(),
            ],
            node_leader: values[8].try_into().unwrap(),
            ranks_on_node: values[9].try_into().unwrap(),
        }
    }

    pub fn add(&mut self, other: &Self) {
        for (left, right) in self
            .counts
            .iter_mut()
            .flatten()
            .zip(other.counts.iter().flatten())
        {
            *left = left
                .checked_add(*right)
                .expect("communication counter overflow");
        }
    }
}

struct Recorder {
    same_node: Vec<bool>,
    report: Report,
}

impl Recorder {
    fn new(rank: Rank, leaders: &[Rank]) -> Self {
        let node_leader = leaders[rank as usize];
        let same_node: Vec<_> = leaders
            .iter()
            .map(|&leader| leader == node_leader)
            .collect();
        let ranks_on_node = same_node.iter().filter(|&&local| local).count() as Rank;
        Self {
            same_node,
            report: Report {
                node_leader,
                ranks_on_node,
                ..Default::default()
            },
        }
    }

    fn record(&mut self, kind: SendKind, destination: Rank, bytes: u64) {
        let offset = if self.same_node[destination as usize] {
            0
        } else {
            2
        };
        let counts = &mut self.report.counts[kind as usize];
        counts[offset] = counts[offset]
            .checked_add(1)
            .expect("message counter overflow");
        counts[offset + 1] = counts[offset + 1]
            .checked_add(bytes)
            .expect("byte counter overflow");
    }
}

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

/// Discover topology before the search window. Node IDs are communicator ranks
/// of shared-memory group leaders, not inferred from rank ordering or hostnames.
pub fn initialize<C: Communicator>(communicator: &C) {
    let shared = communicator.split_shared(communicator.rank());
    let mut leader = communicator.rank();
    shared.process_at_rank(0).broadcast_into(&mut leader);
    let mut leaders = vec![0; communicator.size() as usize];
    communicator.all_gather_into(&leader, &mut leaders[..]);
    RECORDER.with(|recorder| {
        *recorder.borrow_mut() = Some(Recorder::new(communicator.rank(), &leaders));
    });
}

pub fn enabled() -> bool {
    RECORDER.with(|recorder| recorder.borrow().is_some())
}

/// MPI datatype payload size, excluding holes/padding and protocol headers.
pub fn datatype_bytes<D: mpi::datatype::Datatype>(datatype: &D) -> u64 {
    let mut size = 0;
    let result = unsafe { mpi::ffi::MPI_Type_size(datatype.as_raw(), &mut size) };
    assert_eq!(result, mpi::ffi::MPI_SUCCESS as i32, "MPI_Type_size failed");
    u64::try_from(size).expect("negative MPI datatype size")
}

pub fn record_buffer<B: Buffer + ?Sized>(kind: SendKind, destination: Rank, buffer: &B) {
    RECORDER.with(|recorder| {
        if let Some(recorder) = recorder.borrow_mut().as_mut() {
            let bytes = datatype_bytes(&buffer.as_datatype())
                .checked_mul(u64::try_from(buffer.count()).expect("negative MPI count"))
                .expect("message size overflow");
            recorder.record(kind, destination, bytes);
        }
    });
}

/// Hot search-node sends use a datatype size cached when the communicator is built.
pub fn record_bytes(kind: SendKind, destination: Rank, bytes: u64) {
    RECORDER.with(|recorder| {
        if let Some(recorder) = recorder.borrow_mut().as_mut() {
            recorder.record(kind, destination, bytes);
        }
    });
}

pub fn finish() -> Report {
    RECORDER.with(|recorder| {
        recorder
            .borrow_mut()
            .take()
            .expect("communication recorder not initialized")
            .report
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locality_uses_actual_mapping_and_counts_zero_byte_messages() {
        // Deliberately interleaved placement: rank order is not node order.
        let mut recorder = Recorder::new(2, &[0, 1, 0, 1]);
        recorder.record(SendKind::Buffered, 0, 17);
        recorder.record(SendKind::Buffered, 2, 0);
        recorder.record(SendKind::Buffered, 1, 31);
        recorder.record(SendKind::Standard, 3, 8);
        assert_eq!(recorder.report.counts, [[2, 17, 1, 31], [0, 0, 1, 8]]);
        assert_eq!(recorder.report.ranks_on_node, 2);
        assert_eq!(recorder.report.node_leader, 0);
    }

    #[test]
    fn packing_and_aggregation_preserve_large_integer_counts() {
        let a = Report {
            counts: [[1 << 54, (1 << 54) + 1, 3, 19], [1, 0, 2, 4]],
            node_leader: 3,
            ranks_on_node: 6,
        };
        let b = Report::unpack(&a.pack());
        assert_eq!(a.counts, b.counts);
        assert_eq!(b.node_leader, 3);
        let mut sum = Report::default();
        sum.add(&a);
        sum.add(&b);
        assert_eq!(sum.counts[0][1], (1 << 55) + 2);
    }

    #[test]
    fn disabled_recorder_does_not_collect() {
        RECORDER.with(|recorder| *recorder.borrow_mut() = None);
        assert!(!enabled());
        record_bytes(SendKind::Buffered, 900, 128);
        assert!(!enabled());
    }
}
