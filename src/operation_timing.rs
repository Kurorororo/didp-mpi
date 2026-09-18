//! HAC operation timing configuration and MPI CSV reporting.

use linked_hash_map::LinkedHashMap;
use yaml_rust::Yaml;

/// Time one MPI API call. The non-instrumented build evaluates only the call.
#[macro_export]
macro_rules! timed_mpi {
    (MpiBsend, $destination:ident.buffered_send_with_tag($buffer:expr, $tag:expr $(,)?)) => {{
        let mpi_buffer = $buffer;
        #[cfg(feature = "operation-timing")]
        $crate::communication_statistics::record_buffer(
            $crate::communication_statistics::SendKind::Buffered,
            $destination.destination_rank(), mpi_buffer,
        );
        $crate::timed_mpi!(@call MpiBsend, $destination.buffered_send_with_tag(mpi_buffer, $tag))
    }};
    (MpiSend, $destination:ident.send_with_tag($buffer:expr, $tag:expr $(,)?)) => {{
        let mpi_buffer = $buffer;
        #[cfg(feature = "operation-timing")]
        $crate::communication_statistics::record_buffer(
            $crate::communication_statistics::SendKind::Standard,
            $destination.destination_rank(), mpi_buffer,
        );
        $crate::timed_mpi!(@call MpiSend, $destination.send_with_tag(mpi_buffer, $tag))
    }};
    (MpiBsend, $destination:ident.buffered_send_with_tag($buffer:expr, $tag:expr $(,)?), payload_bytes = $bytes:expr) => {{
        #[cfg(feature = "operation-timing")]
        $crate::communication_statistics::record_bytes(
            $crate::communication_statistics::SendKind::Buffered,
            $destination.destination_rank(), $bytes,
        );
        $crate::timed_mpi!(@call MpiBsend, $destination.buffered_send_with_tag($buffer, $tag))
    }};
    (MpiBsend, $call:expr) => {
        compile_error!("buffered sends must expose destination and buffer for communication accounting")
    };
    (MpiSend, $call:expr) => {
        compile_error!("standard sends must expose destination and buffer for communication accounting")
    };
    ($operation:ident, $call:expr) => {{
        $crate::timed_mpi!(@call $operation, $call)
    }};
    (@call $operation:ident, $call:expr) => {{
        #[cfg(feature = "operation-timing")]
        let _mpi_timer = dypdl_heuristic_search::operation_timing::Timer::start(
            dypdl_heuristic_search::operation_timing::Operation::$operation,
            1,
        );
        $call
    }};
}

/// Time and classify a nonblocking probe without changing its return value.
#[macro_export]
macro_rules! timed_mpi_probe {
    ($call:expr) => {{
        #[cfg(feature = "operation-timing")]
        let mut mpi_timer = dypdl_heuristic_search::operation_timing::Timer::start(
            dypdl_heuristic_search::operation_timing::Operation::MpiIprobeMiss,
            1,
        );
        let result = $call;
        #[cfg(feature = "operation-timing")]
        mpi_timer.probe_result(result.is_some());
        result
    }};
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationTimingParameters {
    pub level: u8,
    pub sample_interval: u64,
}

impl OperationTimingParameters {
    pub fn load_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> Self {
        for (old, replacement) in [
            ("operation_timing", "operation_timing_level"),
            ("operation_timing_sample_interval", "timing_sample_interval"),
            ("mpi_timing_sample_interval", "timing_sample_interval"),
            ("mpi_send_timing_sample_interval", "timing_sample_interval"),
            (
                "mpi_send_timing",
                "operation_timing_level; specialized send diagnostics are no longer collected",
            ),
        ] {
            assert!(
                !map.contains_key(&Yaml::String(old.into())),
                "{old} has been removed; use {replacement}"
            );
        }
        Self {
            level: map
                .get(&Yaml::String("operation_timing_level".into()))
                .map(|v| {
                    v.as_i64()
                        .filter(|v| (0..=2).contains(v))
                        .expect("operation_timing_level must be 0, 1, or 2")
                        as u8
                })
                .unwrap_or(0),
            sample_interval: map
                .get(&Yaml::String("timing_sample_interval".into()))
                .map(|v| {
                    v.as_i64()
                        .filter(|v| *v > 0)
                        .expect("timing_sample_interval must be a positive integer")
                        as u64
                })
                .unwrap_or(1),
        }
    }

    pub fn enabled(&self) -> bool {
        self.level > 0
    }

    pub fn validate(&self) {
        assert!(self.level <= 2, "operation_timing_level must be 0, 1, or 2");
        assert!(
            self.sample_interval > 0,
            "timing sample interval must be positive"
        );
        assert!(
            !self.enabled() || cfg!(feature = "operation-timing"),
            "operation timing requires a build with --features operation-timing"
        );
    }
}

#[cfg(feature = "operation-timing")]
pub use dypdl_heuristic_search::operation_timing::start_search;
#[cfg(feature = "operation-timing")]
pub use reporting::finish_and_dump;

#[cfg(feature = "operation-timing")]
mod reporting {
    use crate::communication_statistics::{self, Report as CommunicationReport};
    use dypdl_heuristic_search::operation_timing::{self, Operation, Recording};
    use mpi::traits::*;
    use std::fs::File;
    use std::io::{self, BufWriter, Write};

    // Three exact counters and a corrected estimate (IEEE bits; NaN = unavailable).
    const FIELDS: usize = 4;
    const PACKED_LEN: usize = FIELDS * Operation::ALL.len();
    const COMBINED_LEN: usize = PACKED_LEN + CommunicationReport::PACKED_LEN;
    type Report = [Row; Operation::ALL.len()];

    #[derive(Clone, Copy, Default)]
    struct Row {
        calls: u64,
        timed_calls: u64,
        sampled_ns: u64,
        estimate: f64,
    }

    impl Row {
        fn add(&mut self, row: Self) {
            self.calls += row.calls;
            self.timed_calls += row.timed_calls;
            self.sampled_ns += row.sampled_ns;
            self.estimate += row.estimate;
        }
    }

    fn pack(recording: &Recording) -> [u64; PACKED_LEN] {
        let mut packed = [0; PACKED_LEN];
        for ((m, estimate), fields) in recording
            .measurements
            .iter()
            .zip(recording.estimated_ns)
            .zip(packed.chunks_exact_mut(FIELDS))
        {
            fields.copy_from_slice(&[m.calls, m.timed_calls, m.sampled_ns, estimate.to_bits()]);
        }
        packed
    }

    fn unpack(packed: &[u64]) -> Report {
        assert_eq!(packed.len(), PACKED_LEN);
        std::array::from_fn(|i| {
            let v = &packed[i * FIELDS..];
            Row {
                calls: v[0],
                timed_calls: v[1],
                sampled_ns: v[2],
                estimate: f64::from_bits(v[3]),
            }
        })
    }

    fn ns(value: f64) -> String {
        if value.is_finite() {
            format!("{value:.3}")
        } else {
            String::new()
        }
    }

    fn write_rows<W: Write>(
        writer: &mut W,
        rank: &str,
        report: &Report,
        level: u8,
        interval: u64,
        communication: &CommunicationReport,
    ) -> io::Result<()> {
        for operation in Operation::ALL.into_iter().filter(|op| op.at_level(level)) {
            let row = report[operation as usize];
            let reference = operation == Operation::SearchTotal;
            let accounting = if reference {
                "reference"
            } else if !row.estimate.is_finite() {
                "unavailable"
            } else if interval == 1 {
                "measured"
            } else {
                "estimated"
            };
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                rank,
                operation.name(),
                row.calls,
                row.timed_calls,
                row.sampled_ns,
                ns(row.estimate),
                if row.calls > 0 {
                    ns(row.estimate / row.calls as f64)
                } else {
                    String::new()
                },
                if reference || operation == Operation::NonMpiSearch {
                    1
                } else {
                    interval
                },
                accounting,
                if rank == "all" {
                    String::new()
                } else {
                    communication.node_leader.to_string()
                },
                if rank == "all" {
                    String::new()
                } else {
                    communication.ranks_on_node.to_string()
                },
                match operation {
                    Operation::MpiBsend | Operation::MpiSend => {
                        let i = usize::from(operation == Operation::MpiSend);
                        communication.counts[i]
                            .iter()
                            .map(u64::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    }
                    _ => ",,,".into(),
                }
            )?;
        }
        Ok(())
    }

    /// Every rank stops recording; rank zero writes one CSV containing all ranks
    /// followed by their aggregate. No report collection is included in timing.
    pub fn finish_and_dump<C: Communicator>(communicator: &C) -> io::Result<()> {
        let recording = operation_timing::finish_recording();
        let communication = communication_statistics::finish();
        for (i, op) in [Operation::MpiBsend, Operation::MpiSend].iter().enumerate() {
            assert_eq!(
                recording.measurements[*op as usize].calls,
                communication.counts[i][0] + communication.counts[i][2],
                "point-to-point send was not classified"
            );
        }
        let mut local = [0; COMBINED_LEN];
        local[..PACKED_LEN].copy_from_slice(&pack(&recording));
        local[PACKED_LEN..].copy_from_slice(&communication.pack());
        let root = communicator.process_at_rank(0);
        if communicator.rank() != 0 {
            root.gather_into(&local[..]);
            return Ok(());
        }
        let mut gathered = vec![0; COMBINED_LEN * communicator.size() as usize];
        root.gather_into_root(&local[..], &mut gathered[..]);
        let mut total = [Row::default(); Operation::ALL.len()];
        let mut communication_total = CommunicationReport::default();
        let mut file = BufWriter::new(File::create("timing.csv")?);
        writeln!(file, "rank,operation,calls,timed_calls,sampled_ns,estimated_total_ns,mean_ns,sample_interval,accounting,node_leader_rank,ranks_on_node,intra_node_messages,intra_node_payload_bytes,inter_node_messages,inter_node_payload_bytes")?;
        for (rank, packed) in gathered.chunks_exact(COMBINED_LEN).enumerate() {
            let report = unpack(&packed[..PACKED_LEN]);
            let communication = CommunicationReport::unpack(&packed[PACKED_LEN..]);
            communication_total.add(&communication);
            for (total, row) in total.iter_mut().zip(&report) {
                total.add(*row);
            }
            write_rows(
                &mut file,
                &rank.to_string(),
                &report,
                recording.level,
                recording.sample_interval,
                &communication,
            )?;
        }
        write_rows(
            &mut file,
            "all",
            &total,
            recording.level,
            recording.sample_interval,
            &communication_total,
        )?;
        file.flush()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use dypdl_heuristic_search::operation_timing::Measurement;
        #[test]
        fn packing_preserves_corrected_and_missing_estimates() {
            let mut recording = Recording {
                measurements: [Measurement::default(); Operation::ALL.len()],
                estimated_ns: [0.0; Operation::ALL.len()],
                level: 2,
                sample_interval: 7,
            };
            recording.measurements[0] = Measurement {
                calls: 3,
                timed_calls: 1,
                sampled_ns: 20,
                ..Default::default()
            };
            recording.estimated_ns[0] = -10.0;
            recording.estimated_ns[1] = f64::NAN;
            let report = unpack(&pack(&recording));
            assert_eq!(
                (report[0].calls, report[0].timed_calls, report[0].sampled_ns),
                (3, 1, 20)
            );
            assert_eq!(report[0].estimate, -10.0);
            assert!(report[1].estimate.is_nan());
        }
        #[test]
        fn aggregation_sums_rank_estimates_and_preserves_missing() {
            let mut total = Row::default();
            total.add(Row {
                calls: 3,
                estimate: 60.0,
                ..Default::default()
            });
            total.add(Row {
                calls: 1,
                estimate: 100.0,
                ..Default::default()
            });
            assert_eq!(total.calls, 4);
            assert_eq!(total.estimate, 160.0);
            total.add(Row {
                estimate: f64::NAN,
                ..Default::default()
            });
            assert!(total.estimate.is_nan());
        }
        #[test]
        fn compact_rows_preserve_unavailable_and_negative_values() {
            let mut report = [Row::default(); Operation::ALL.len()];
            report[Operation::NonMpiSearch as usize] = Row {
                calls: 1,
                timed_calls: 1,
                sampled_ns: 10,
                estimate: -30.0,
            };
            report[Operation::MpiIprobeHit as usize] = Row {
                calls: 2,
                estimate: f64::NAN,
                ..Default::default()
            };
            let mut bytes = vec![];
            write_rows(
                &mut bytes,
                "0",
                &report,
                1,
                100,
                &CommunicationReport::default(),
            )
            .unwrap();
            write_rows(
                &mut bytes,
                "all",
                &report,
                1,
                100,
                &CommunicationReport::default(),
            )
            .unwrap();
            let csv = String::from_utf8(bytes).unwrap();
            assert_eq!(csv.lines().count(), 18);
            assert!(csv.lines().all(|line| line.split(',').count() == 15));
            assert!(!csv.contains("NaN"));
            assert!(csv.contains("0,non_mpi_search,1,1,10,-30.000,-30.000,1,estimated"));
            assert!(csv.contains("all,MPI_Iprobe_hit,2,0,0,,,100,unavailable"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yaml_rust::YamlLoader;
    fn load(yaml: &str) -> OperationTimingParameters {
        let yaml = YamlLoader::load_from_str(yaml).unwrap();
        OperationTimingParameters::load_from_map(yaml[0].as_hash().unwrap())
    }
    #[test]
    fn defaults_and_two_options() {
        assert_eq!(
            load("{}"),
            OperationTimingParameters {
                level: 0,
                sample_interval: 1
            }
        );
        load("{}").validate();
        assert_eq!(
            load("operation_timing_level: 2\ntiming_sample_interval: 100"),
            OperationTimingParameters {
                level: 2,
                sample_interval: 100
            }
        );
    }
    #[test]
    fn invalid_and_retired_options_are_not_silently_ignored() {
        for key in [
            "operation_timing",
            "operation_timing_sample_interval",
            "mpi_timing_sample_interval",
            "mpi_send_timing",
            "mpi_send_timing_sample_interval",
        ] {
            assert!(std::panic::catch_unwind(|| load(&format!("{key}: 1"))).is_err());
        }
        for value in ["-1", "3", "1.5", "true", "'1'"] {
            assert!(
                std::panic::catch_unwind(|| load(&format!("operation_timing_level: {value}")))
                    .is_err()
            );
        }
        for value in ["0", "-1", "1.5", "true", "'100'"] {
            assert!(
                std::panic::catch_unwind(|| load(&format!("timing_sample_interval: {value}")))
                    .is_err()
            );
        }
    }
    #[cfg(not(feature = "operation-timing"))]
    #[test]
    #[should_panic(expected = "requires a build with --features operation-timing")]
    fn rejects_unavailable_instrumentation() {
        load("operation_timing_level: 1").validate();
    }
    #[cfg(feature = "operation-timing")]
    #[test]
    fn accepts_available_instrumentation() {
        load("operation_timing_level: 1").validate();
    }
}
