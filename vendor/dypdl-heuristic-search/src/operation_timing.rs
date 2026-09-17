//! Opt-in operation counters and sampled elapsed timings for the current thread.
//!
//! Parent timers are inclusive of nested timers and their bookkeeping. Timings
//! use wall-clock elapsed time, not CPU time. No output occurs on the hot path.

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
}

impl Operation {
    /// All operations, ordered by discriminant.
    pub const ALL: [Self; 13] = [
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
        }
    }
}

/// Exact counts and sampled time for one operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Measurement {
    /// Number of calls, including calls that were not timed.
    pub calls: u64,
    /// Number of timed calls.
    pub timed_calls: u64,
    /// Sum of elapsed nanoseconds for timed calls only.
    pub sampled_ns: u64,
    /// Sum of the input sizes across all calls.
    pub size_sum: u64,
    /// Maximum input size.
    pub max_size: u64,
    /// Number of calls with zero input size (e.g., empty heap pops).
    pub zero_size_calls: u64,
}

impl Measurement {
    /// Estimated total nanoseconds, using this rank's own sample mean.
    pub fn estimated_ns(self) -> f64 {
        if self.timed_calls == 0 {
            0.0
        } else {
            self.sampled_ns as f64 / self.timed_calls as f64 * self.calls as f64
        }
    }
}

/// One thread's completed measurements.
pub type Measurements = [Measurement; Operation::ALL.len()];

struct Recorder {
    sample_interval: u64,
    measurements: Measurements,
}

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

/// Enable recording on this thread. Call before constructing the solver.
/// Panics for a zero interval or an already active recording session.
pub fn start(sample_interval: u64) {
    assert!(
        sample_interval > 0,
        "operation timing sample interval must be positive"
    );
    RECORDER.with(|recorder| {
        let mut recorder = recorder.borrow_mut();
        assert!(recorder.is_none(), "operation timing is already active");
        *recorder = Some(Recorder {
            sample_interval,
            measurements: [Measurement::default(); Operation::ALL.len()],
        });
    });
}

/// Stop recording and return measurements. All active timers must have ended.
pub fn finish() -> Measurements {
    RECORDER.with(|recorder| {
        recorder
            .borrow_mut()
            .take()
            .expect("operation timing is not active")
            .measurements
    })
}

/// Records a call and, when sampled, its duration through scope exit.
/// Do not retain this guard beyond the operation being measured.
pub struct Timer {
    operation: Operation,
    start: Option<Instant>,
}

impl Timer {
    /// Input size is the pre-operation heap length, signature count, or label
    /// count, depending on the operation. Comparisons use size one.
    #[inline]
    pub fn start(operation: Operation, size: usize) -> Self {
        let sampled = RECORDER.with(|recorder| {
            let mut recorder = recorder.borrow_mut();
            let Some(recorder) = recorder.as_mut() else {
                return false;
            };
            let measurement = &mut recorder.measurements[operation as usize];
            let sampled = measurement.calls % recorder.sample_interval == 0;
            measurement.calls += 1;
            measurement.size_sum += size as u64;
            measurement.max_size = measurement.max_size.max(size as u64);
            measurement.zero_size_calls += u64::from(size == 0);
            sampled
        });
        Self {
            operation,
            start: sampled.then(Instant::now),
        }
    }
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = u64::try_from(start.elapsed().as_nanos())
                .expect("operation duration exceeds u64 nanoseconds");
            RECORDER.with(|recorder| {
                if let Some(recorder) = recorder.borrow_mut().as_mut() {
                    let measurement = &mut recorder.measurements[self.operation as usize];
                    measurement.timed_calls += 1;
                    measurement.sampled_ns += elapsed;
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_and_sampled_sessions() {
        drop(Timer::start(Operation::RegistryLookup, 99));
        start(3);
        for size in 0..8 {
            drop(Timer::start(Operation::RegistryLookup, size));
        }
        let measurements = finish();
        let measurement = measurements[Operation::RegistryLookup as usize];
        assert_eq!(measurement.calls, 8);
        assert_eq!(measurement.timed_calls, 3);
        assert_eq!(measurement.size_sum, 28);
        assert_eq!(measurement.max_size, 7);
        assert_eq!(measurement.zero_size_calls, 1);
        assert_eq!(
            measurements[Operation::DominanceScan as usize],
            Measurement::default()
        );
        start(1);
        assert_eq!(finish(), [Measurement::default(); Operation::ALL.len()]);
    }

    #[test]
    fn nested_timers_and_early_returns() {
        fn operation() {
            let _parent = Timer::start(Operation::DominanceScan, 2);
            let _child = Timer::start(Operation::DominanceCompare, 1);
        }
        start(1);
        operation();
        let measurements = finish();
        let parent = measurements[Operation::DominanceScan as usize];
        let child = measurements[Operation::DominanceCompare as usize];
        assert_eq!(parent.calls, 1);
        assert_eq!(parent.timed_calls, 1);
        assert_eq!(child.calls, 1);
        assert_eq!(child.timed_calls, 1);
        assert!(parent.sampled_ns >= child.sampled_ns);
    }

    #[test]
    fn empty_estimate_is_zero() {
        assert_eq!(Measurement::default().estimated_ns(), 0.0);
    }

    #[test]
    #[should_panic(expected = "sample interval must be positive")]
    fn rejects_zero_interval() {
        start(0);
    }
}
