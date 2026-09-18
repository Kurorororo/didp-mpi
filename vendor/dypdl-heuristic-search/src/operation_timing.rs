//! Sampled operation timing within the HAC search window.
//! Each category uses one interval. Logical parent/child counts permit exclusive
//! estimates without timing every nested call. SearchTotal is always measured.

use std::cell::RefCell;
use std::time::Instant;

/// Instrumented operation, in the stable order used by MPI reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Operation {
    /// Push to the primary heap.
    HeapPrimaryPush,
    /// Raw pop from a nonempty primary heap.
    HeapPrimaryPop,
    /// Raw pop from an empty primary heap.
    HeapPrimaryEmptyPop,
    /// Clear the primary heap after bound pruning.
    HeapPrimaryClear,
    /// Push to a depth-specific heap.
    HeapLayeredPush,
    /// Raw pop from a nonempty depth-specific heap.
    HeapLayeredPop,
    /// Raw pop from an empty depth-specific heap.
    HeapLayeredEmptyPop,
    /// Clear a depth-specific heap after bound pruning.
    HeapLayeredClear,
    /// Entire registry insert, including lookup and dominance handling.
    RegistryInsert,
    /// Entire registry insert_with, including the supplied constructor.
    RegistryInsertWith,
    /// Hash-map entry lookup in either insertion path.
    RegistryLookup,
    /// Scan a signature's labels and remove dominated labels.
    DominanceScan,
    /// One state-metadata dominance comparison, excluding cost comparison.
    DominanceCompare,
    /// Complete HAC search call (always timed).
    SearchTotal,
    /// Periodic memory monitoring in the HAC loop.
    PhaseMonitoring,
    /// Drain incoming messages, including processing them.
    PhaseMessages,
    /// Timeout and termination checks before selecting work.
    PhaseControl,
    /// Select a node, including depth traversal and stale-node rejection.
    PhaseSelection,
    /// Expand one node, including expansion-statistics recording.
    PhaseExpansion,
    /// Send all remote successors of one expansion.
    PhaseSending,
    /// Enqueue all local successors of one expansion.
    PhaseEnqueue,
    /// Termination initiation after an unsuccessful selection.
    PhaseNoWork,
    /// Final HAC barrier.
    PhaseBarrier,
    /// Final solution/statistics collection and result bookkeeping.
    PhaseFinalize,
    /// Advance the applicable-transition iterator, including its final None.
    TransitionNext,
    /// Generate a successor state and its cost, including constraint checks.
    SuccessorGeneration,
    /// Check a successor for a goal and process any solution found.
    GoalCheck,
    /// Compute the successor's ownership hash.
    OwnershipHash,
    /// Local successor evaluator, including registry insertion.
    LocalSuccessorEvaluation,
    /// Remote successor evaluator, including message-node construction.
    RemoteSuccessorEvaluation,
    /// Evaluate the dual bound in the local successor path.
    HeuristicLocal,
    /// Evaluate the dual bound in the outgoing successor path.
    HeuristicRemote,
    /// Dispatch and process one incoming message of any tag.
    MessageDispatch,
    /// Serialize a node and its depth, excluding the timestamp and MPI call.
    NodeSerialize,
    /// Deserialize an accepted received node and its depth.
    NodeDeserialize,
    /// Search-window work outside MPI (or outside detailed scopes at level 2).
    NonMpiSearch,
    /// All buffered sends, including control and solution messages.
    MpiBsend,
    /// Blocking standard sends.
    MpiSend,
    /// Blocking receives, including control and solution messages.
    MpiRecv,
    /// Immediate probes that found a message.
    MpiIprobeHit,
    /// Immediate probes that found no message.
    MpiIprobeMiss,
    /// Barrier calls.
    MpiBarrier,
    /// Gather calls on both root and non-root ranks.
    MpiGather,
}

impl Operation {
    /// All operations, ordered by discriminant.
    pub const ALL: [Self; 43] = [
        Self::HeapPrimaryPush,
        Self::HeapPrimaryPop,
        Self::HeapPrimaryEmptyPop,
        Self::HeapPrimaryClear,
        Self::HeapLayeredPush,
        Self::HeapLayeredPop,
        Self::HeapLayeredEmptyPop,
        Self::HeapLayeredClear,
        Self::RegistryInsert,
        Self::RegistryInsertWith,
        Self::RegistryLookup,
        Self::DominanceScan,
        Self::DominanceCompare,
        Self::SearchTotal,
        Self::PhaseMonitoring,
        Self::PhaseMessages,
        Self::PhaseControl,
        Self::PhaseSelection,
        Self::PhaseExpansion,
        Self::PhaseSending,
        Self::PhaseEnqueue,
        Self::PhaseNoWork,
        Self::PhaseBarrier,
        Self::PhaseFinalize,
        Self::TransitionNext,
        Self::SuccessorGeneration,
        Self::GoalCheck,
        Self::OwnershipHash,
        Self::LocalSuccessorEvaluation,
        Self::RemoteSuccessorEvaluation,
        Self::HeuristicLocal,
        Self::HeuristicRemote,
        Self::MessageDispatch,
        Self::NodeSerialize,
        Self::NodeDeserialize,
        Self::NonMpiSearch,
        Self::MpiBsend,
        Self::MpiSend,
        Self::MpiRecv,
        Self::MpiIprobeHit,
        Self::MpiIprobeMiss,
        Self::MpiBarrier,
        Self::MpiGather,
    ];

    /// CSV operation name.
    pub fn name(self) -> &'static str {
        match self {
            Self::HeapPrimaryPush => "heap_primary_push",
            Self::HeapPrimaryPop => "heap_primary_pop",
            Self::HeapPrimaryEmptyPop => "heap_primary_empty_pop",
            Self::HeapPrimaryClear => "heap_primary_clear",
            Self::HeapLayeredPush => "heap_layered_push",
            Self::HeapLayeredPop => "heap_layered_pop",
            Self::HeapLayeredEmptyPop => "heap_layered_empty_pop",
            Self::HeapLayeredClear => "heap_layered_clear",
            Self::RegistryInsert => "registry_insert",
            Self::RegistryInsertWith => "registry_insert_with",
            Self::RegistryLookup => "registry_lookup",
            Self::DominanceScan => "dominance_scan",
            Self::DominanceCompare => "dominance_compare",
            Self::SearchTotal => "search_total",
            Self::PhaseMonitoring => "monitoring",
            Self::PhaseMessages => "message_handling",
            Self::PhaseControl => "control",
            Self::PhaseSelection => "node_selection",
            Self::PhaseExpansion => "expansion",
            Self::PhaseSending => "sending",
            Self::PhaseEnqueue => "local_enqueue",
            Self::PhaseNoWork => "no_work_control",
            Self::PhaseBarrier => "final_barrier",
            Self::PhaseFinalize => "finalize",
            Self::TransitionNext => "transition_next",
            Self::SuccessorGeneration => "successor_generation",
            Self::GoalCheck => "goal_check",
            Self::OwnershipHash => "ownership_hash",
            Self::LocalSuccessorEvaluation => "local_successor_evaluation",
            Self::RemoteSuccessorEvaluation => "remote_successor_evaluation",
            Self::HeuristicLocal => "heuristic_local",
            Self::HeuristicRemote => "heuristic_remote",
            Self::MessageDispatch => "message_dispatch",
            Self::NodeSerialize => "node_serialize",
            Self::NodeDeserialize => "node_deserialize",
            Self::NonMpiSearch => "non_mpi_search",
            Self::MpiBsend => "MPI_Bsend",
            Self::MpiSend => "MPI_Send",
            Self::MpiRecv => "MPI_Recv",
            Self::MpiIprobeHit => "MPI_Iprobe_hit",
            Self::MpiIprobeMiss => "MPI_Iprobe_miss",
            Self::MpiBarrier => "MPI_Barrier",
            Self::MpiGather => "MPI_Gather",
        }
    }

    /// MPI communication calls are enabled at both active levels.
    pub fn is_mpi(self) -> bool {
        (self as usize) >= Self::MpiBsend as usize
    }

    pub fn at_level(self, level: u8) -> bool {
        level >= 2 || self.is_mpi() || matches!(self, Self::SearchTotal | Self::NonMpiSearch)
    }
}

/// Exact counts and measured time for an operation. Size counters support
/// instrumentation checks; the compact CSV exports timing fields only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Measurement {
    pub calls: u64,
    pub timed_calls: u64,
    /// Time charged to sampled scopes, excluding other sampled scopes. Skipped
    /// descendants remain in this raw duration; use estimates for breakdowns.
    pub sampled_ns: u64,
    pub size_sum: u64,
    pub max_size: u64,
    pub zero_size_calls: u64,
}

pub type Measurements = [Measurement; Operation::ALL.len()];

pub struct Recording {
    pub measurements: Measurements,
    /// Exclusive estimates; NaN means an observed category lacked samples.
    pub estimated_ns: [f64; Operation::ALL.len()],
    pub level: u8,
    pub sample_interval: u64,
}

#[derive(Clone, Copy, Default)]
struct Edge {
    calls: u64,
    timed_calls: u64,
    inclusive_ns: u64,
}

struct Frame {
    operation: Operation,
    start_ns: Option<u64>,
}

struct Recorder {
    level: u8,
    sample_interval: u64,
    measurements: Measurements,
    inclusive_ns: [u64; Operation::ALL.len()],
    // Direct logical parent/child relationships, including unsampled scopes.
    edges: [[Edge; Operation::ALL.len()]; Operation::ALL.len()],
    frames: Vec<Frame>,
    origin: Option<Instant>,
    last_ns: u64,
    probe_calls: u64,
}

impl Recorder {
    fn new(level: u8, sample_interval: u64) -> Self {
        Self {
            level,
            sample_interval,
            measurements: [Measurement::default(); Operation::ALL.len()],
            inclusive_ns: [0; Operation::ALL.len()],
            edges: [[Edge::default(); Operation::ALL.len()]; Operation::ALL.len()],
            frames: Vec::with_capacity(16),
            origin: None,
            last_ns: 0,
            probe_calls: 0,
        }
    }

    fn now(&self) -> u64 {
        u64::try_from(self.origin.unwrap().elapsed().as_nanos())
            .expect("search duration exceeds u64 nanoseconds")
    }

    // Only sampled boundaries call the clock. All timestamps use one origin,
    // preserving an exact partition of raw time without rounding each segment.
    fn tick_at(&mut self, now: u64) {
        if let Some(frame) = self.frames.iter().rev().find(|f| f.start_ns.is_some()) {
            self.measurements[frame.operation as usize].sampled_ns += now - self.last_ns;
        }
        self.last_ns = now;
    }

    fn count(&mut self, operation: Operation, size: usize, sampled: bool) {
        let m = &mut self.measurements[operation as usize];
        m.calls += 1;
        m.timed_calls += u64::from(sampled);
        m.size_sum += size as u64;
        m.max_size = m.max_size.max(size as u64);
        m.zero_size_calls += u64::from(size == 0);
    }

    fn enter(&mut self, operation: Operation, size: usize) -> bool {
        if operation == Operation::SearchTotal {
            assert!(self.frames.is_empty(), "search windows must not overlap");
            self.origin = Some(Instant::now());
            self.last_ns = 0;
            self.frames.push(Frame {
                operation: Operation::NonMpiSearch,
                start_ns: Some(0),
            });
            self.count(Operation::SearchTotal, 1, true);
            self.count(Operation::NonMpiSearch, 1, true);
            return true;
        }
        if self.frames.is_empty() || !operation.at_level(self.level) {
            return false;
        }
        let calls = if operation == Operation::MpiIprobeMiss {
            let calls = self.probe_calls;
            self.probe_calls += 1;
            calls
        } else {
            self.measurements[operation as usize].calls
        };
        let sampled = calls % self.sample_interval == 0;
        self.count(operation, size, sampled);
        let start_ns = sampled.then(|| {
            let now = self.now();
            self.tick_at(now);
            now
        });
        self.frames.push(Frame {
            operation,
            start_ns,
        });
        true
    }

    fn leave(&mut self, operation: Operation) {
        let expected = if operation == Operation::SearchTotal {
            Operation::NonMpiSearch
        } else {
            operation
        };
        assert_eq!(
            self.frames.last().map(|f| f.operation),
            Some(expected),
            "timers must end in nesting order"
        );
        // A skipped scope, even one with sampled descendants, reads no clock.
        let elapsed = self.frames.last().unwrap().start_ns.map(|start| {
            let now = self.now();
            self.tick_at(now);
            now - start
        });
        self.frames.pop();
        if let Some(elapsed) = elapsed {
            self.inclusive_ns[expected as usize] += elapsed;
        }
        if let Some(parent) = self.frames.last() {
            let edge = &mut self.edges[parent.operation as usize][expected as usize];
            edge.calls += 1;
            if let Some(elapsed) = elapsed {
                edge.timed_calls += 1;
                edge.inclusive_ns += elapsed;
            }
        } else {
            assert_eq!(operation, Operation::SearchTotal);
            self.measurements[Operation::SearchTotal as usize].sampled_ns += self.last_ns;
            self.inclusive_ns[Operation::SearchTotal as usize] += self.last_ns;
            self.origin = None;
        }
    }

    fn estimates(&self) -> [f64; Operation::ALL.len()] {
        let means: [f64; Operation::ALL.len()] = std::array::from_fn(|i| {
            let m = self.measurements[i];
            if m.timed_calls > 0 {
                self.inclusive_ns[i] as f64 / m.timed_calls as f64
            } else if m.calls == 0 {
                0.0
            } else {
                f64::NAN
            }
        });
        let mut estimates = std::array::from_fn(|i| {
            let m = self.measurements[i];
            // Never multiply an unavailable mean by zero.
            self.inclusive_ns[i] as f64
                + if m.calls > m.timed_calls {
                    means[i] * (m.calls - m.timed_calls) as f64
                } else {
                    0.0
                }
        });
        for (parent, edges) in self.edges.iter().enumerate() {
            for (child, edge) in edges.iter().enumerate() {
                estimates[parent] -= edge.inclusive_ns as f64
                    + if edge.calls > edge.timed_calls {
                        means[child] * (edge.calls - edge.timed_calls) as f64
                    } else {
                        0.0
                    };
            }
        }
        estimates
    }
}

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

/// Start a search recording session. Only scopes within SearchTotal are counted.
pub fn start_search(level: u8, sample_interval: u64) {
    assert!(
        (1..=2).contains(&level),
        "active timing level must be 1 or 2"
    );
    assert!(
        sample_interval > 0,
        "timing sample interval must be positive"
    );
    RECORDER.with(|recorder| {
        let mut recorder = recorder.borrow_mut();
        assert!(recorder.is_none(), "operation timing is already active");
        *recorder = Some(Recorder::new(level, sample_interval));
    });
}

pub fn finish_recording() -> Recording {
    RECORDER.with(|recorder| {
        let recorder = recorder
            .borrow_mut()
            .take()
            .expect("operation timing is not active");
        assert!(recorder.frames.is_empty(), "search timer is still active");
        Recording {
            measurements: recorder.measurements,
            estimated_ns: recorder.estimates(),
            level: recorder.level,
            sample_interval: recorder.sample_interval,
        }
    })
}

/// A logical scope, sampled independently by operation and rank. Keep guards
/// nested even when they are unsampled, so estimates exclude direct children.
pub struct Timer {
    operation: Operation,
    active: bool,
}

impl Timer {
    #[inline]
    pub fn start(operation: Operation, size: usize) -> Self {
        let active = RECORDER.with(|recorder| {
            recorder
                .borrow_mut()
                .as_mut()
                .map(|recorder| recorder.enter(operation, size))
                .unwrap_or(false)
        });
        Self { operation, active }
    }

    /// Probes share one pre-call sampling sequence; outcomes are counted and
    /// classified after the call, including probes that were not timed.
    pub fn probe_result(&mut self, hit: bool) {
        if !self.active || !hit {
            return;
        }
        assert_eq!(self.operation, Operation::MpiIprobeMiss);
        RECORDER.with(|recorder| {
            let mut recorder = recorder.borrow_mut();
            let recorder = recorder.as_mut().unwrap();
            let frame = recorder.frames.last_mut().unwrap();
            assert_eq!(frame.operation, Operation::MpiIprobeMiss);
            frame.operation = Operation::MpiIprobeHit;
            let sampled = frame.start_ns.is_some();
            let miss = &mut recorder.measurements[Operation::MpiIprobeMiss as usize];
            miss.calls -= 1;
            miss.timed_calls -= u64::from(sampled);
            miss.size_sum -= 1;
            if miss.calls == 0 {
                miss.max_size = 0;
            }
            recorder.count(Operation::MpiIprobeHit, 1, sampled);
        });
        self.operation = Operation::MpiIprobeHit;
    }
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        if self.active {
            RECORDER.with(|recorder| {
                recorder
                    .borrow_mut()
                    .as_mut()
                    .unwrap()
                    .leave(self.operation)
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_nested_accounting_is_exact() {
        start_search(2, 1);
        // Initialization is outside the search window.
        drop(Timer::start(Operation::HeapPrimaryPush, 9));
        {
            let _search = Timer::start(Operation::SearchTotal, 1);
            let _parent = Timer::start(Operation::RegistryInsert, 1);
            drop(Timer::start(Operation::RegistryLookup, 1));
            drop(Timer::start(Operation::MpiBsend, 1));
        }
        let recording = finish_recording();
        let total = recording.measurements[Operation::SearchTotal as usize].sampled_ns;
        assert!(total > 0);
        assert_eq!(
            recording
                .measurements
                .iter()
                .map(|m| m.sampled_ns)
                .sum::<u64>(),
            total * 2
        );
        for (m, estimate) in recording.measurements.iter().zip(recording.estimated_ns) {
            assert_eq!(m.sampled_ns as f64, estimate);
            assert_eq!(m.calls, m.timed_calls);
        }
        assert_eq!(
            recording.measurements[Operation::HeapPrimaryPush as usize].calls,
            0
        );
    }

    #[test]
    fn all_operations_share_the_interval_and_skipped_scopes_read_no_clock() {
        for level in [1, 2] {
            start_search(level, 3);
            {
                let _search = Timer::start(Operation::SearchTotal, 1);
                for op in [Operation::MpiRecv, Operation::HeapPrimaryPush] {
                    for i in 0..8 {
                        let before = RECORDER.with(|r| r.borrow().as_ref().unwrap().last_ns);
                        drop(Timer::start(op, 1));
                        let after = RECORDER.with(|r| r.borrow().as_ref().unwrap().last_ns);
                        if i % 3 != 0 || !op.at_level(level) {
                            assert_eq!(before, after);
                        }
                    }
                }
            }
            let recording = finish_recording();
            let recv = recording.measurements[Operation::MpiRecv as usize];
            let heap = recording.measurements[Operation::HeapPrimaryPush as usize];
            assert_eq!((recv.calls, recv.timed_calls), (8, 3));
            assert_eq!(
                (heap.calls, heap.timed_calls),
                if level == 2 { (8, 3) } else { (0, 0) }
            );
            let total = recording.measurements[Operation::SearchTotal as usize].sampled_ns as f64;
            assert!((recording.estimated_ns.iter().sum::<f64>() - 2.0 * total).abs() < 0.001);
        }
    }

    #[test]
    fn estimates_subtract_logical_children_even_when_parent_is_unsampled() {
        // Exact synthetic times: root 100, two parent calls of 30 each, two
        // child calls of 10 each. Only the first parent and second child sample.
        let mut r = Recorder::new(2, 2);
        for (op, calls, timed, ns) in [
            (Operation::NonMpiSearch, 1, 1, 100),
            (Operation::RegistryInsert, 2, 1, 30),
            (Operation::RegistryLookup, 2, 1, 10),
        ] {
            r.measurements[op as usize] = Measurement {
                calls,
                timed_calls: timed,
                ..Default::default()
            };
            r.inclusive_ns[op as usize] = ns;
        }
        r.edges[Operation::NonMpiSearch as usize][Operation::RegistryInsert as usize] = Edge {
            calls: 2,
            timed_calls: 1,
            inclusive_ns: 30,
        };
        r.edges[Operation::RegistryInsert as usize][Operation::RegistryLookup as usize] = Edge {
            calls: 2,
            timed_calls: 1,
            inclusive_ns: 10,
        };
        let estimates = r.estimates();
        assert_eq!(estimates[Operation::NonMpiSearch as usize], 40.0);
        assert_eq!(estimates[Operation::RegistryInsert as usize], 40.0);
        assert_eq!(estimates[Operation::RegistryLookup as usize], 20.0);
        assert_eq!(estimates.iter().sum::<f64>(), 100.0);
        // Keep negative residuals visible instead of clamping sampling error.
        r.inclusive_ns[Operation::RegistryLookup as usize] = 100;
        r.edges[Operation::RegistryInsert as usize][Operation::RegistryLookup as usize]
            .inclusive_ns = 100;
        assert!(r.estimates()[Operation::RegistryInsert as usize] < 0.0);
    }

    #[test]
    fn sampled_descendant_of_skipped_parent_preserves_raw_partition() {
        start_search(2, 2);
        {
            let _search = Timer::start(Operation::SearchTotal, 1);
            drop(Timer::start(Operation::RegistryInsert, 1)); // sampled
            {
                let _parent = Timer::start(Operation::RegistryInsert, 1); // skipped
                drop(Timer::start(Operation::RegistryLookup, 1)); // sampled
            }
        }
        let recording = finish_recording();
        let total = recording.measurements[Operation::SearchTotal as usize].sampled_ns;
        assert_eq!(
            recording
                .measurements
                .iter()
                .map(|m| m.sampled_ns)
                .sum::<u64>(),
            total * 2
        );
        assert!((recording.estimated_ns.iter().sum::<f64>() - 2.0 * total as f64).abs() < 0.001);
    }

    #[test]
    fn probes_count_unsampled_outcomes_and_preserve_unknown_estimates() {
        start_search(1, 100);
        {
            let _search = Timer::start(Operation::SearchTotal, 1);
            for hit in [false, true, true, false] {
                let mut probe = Timer::start(Operation::MpiIprobeMiss, 1);
                probe.probe_result(hit);
            }
        }
        let r = finish_recording();
        let hit = r.measurements[Operation::MpiIprobeHit as usize];
        let miss = r.measurements[Operation::MpiIprobeMiss as usize];
        assert_eq!((hit.calls, hit.timed_calls), (2, 0));
        assert_eq!((miss.calls, miss.timed_calls), (2, 1));
        assert!(r.estimated_ns[Operation::MpiIprobeHit as usize].is_nan());
        assert!(r.estimated_ns[Operation::NonMpiSearch as usize].is_nan());
        assert!(r.estimated_ns[Operation::SearchTotal as usize].is_finite());
    }

    #[test]
    fn metadata_and_disabled_recording() {
        drop(Timer::start(Operation::MpiRecv, 1));
        let mut names = std::collections::HashSet::new();
        for (i, op) in Operation::ALL.into_iter().enumerate() {
            assert_eq!(op as usize, i);
            assert!(names.insert(op.name()));
        }
        assert_eq!(Operation::ALL.iter().filter(|op| op.at_level(1)).count(), 9);
        assert_eq!(
            Operation::ALL.iter().filter(|op| op.at_level(2)).count(),
            43
        );
    }
}
