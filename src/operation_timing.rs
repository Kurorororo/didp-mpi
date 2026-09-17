//! HAC operation timing configuration and MPI CSV reporting.

use linked_hash_map::LinkedHashMap;
use yaml_rust::Yaml;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationTimingParameters {
    pub enabled: bool,
    pub sample_interval: u64,
}

impl OperationTimingParameters {
    pub fn load_from_map(map: &LinkedHashMap<Yaml, Yaml>) -> Self {
        let enabled = crate::load_bool_from_map(map, "operation_timing").unwrap_or(false);
        let sample_interval = map
            .get(&Yaml::String("operation_timing_sample_interval".into()))
            .map(|value| {
                value
                    .as_i64()
                    .filter(|value| *value > 0)
                    .expect("operation_timing_sample_interval must be a positive integer")
                    as u64
            })
            .unwrap_or(1);
        Self {
            enabled,
            sample_interval,
        }
    }

    pub fn validate(&self) {
        assert!(
            self.sample_interval > 0,
            "operation timing sample interval must be positive"
        );
        assert!(
            !self.enabled || cfg!(feature = "operation-timing"),
            "operation_timing: true requires a build with --features operation-timing"
        );
    }
}

#[cfg(feature = "operation-timing")]
pub use dypdl_heuristic_search::operation_timing::start;
#[cfg(feature = "operation-timing")]
pub use reporting::finish_and_dump;

#[cfg(feature = "operation-timing")]
mod reporting {
    use dypdl_heuristic_search::operation_timing::{self, Measurement, Measurements, Operation};
    use mpi::traits::*;
    use std::fs::File;
    use std::io::{self, BufWriter, Write};

    const FIELDS: usize = 6;
    const PACKED_LEN: usize = FIELDS * Operation::ALL.len();

    fn pack(measurements: &Measurements) -> [u64; PACKED_LEN] {
        let mut buffer = [0; PACKED_LEN];
        for (measurement, values) in measurements.iter().zip(buffer.chunks_exact_mut(FIELDS)) {
            values.copy_from_slice(&[
                measurement.calls,
                measurement.timed_calls,
                measurement.sampled_ns,
                measurement.size_sum,
                measurement.max_size,
                measurement.zero_size_calls,
            ]);
        }
        buffer
    }

    fn unpack(buffer: &[u64]) -> Measurements {
        assert_eq!(buffer.len(), PACKED_LEN);
        let mut measurements = [Measurement::default(); Operation::ALL.len()];
        for (measurement, values) in measurements.iter_mut().zip(buffer.chunks_exact(FIELDS)) {
            *measurement = Measurement {
                calls: values[0],
                timed_calls: values[1],
                sampled_ns: values[2],
                size_sum: values[3],
                max_size: values[4],
                zero_size_calls: values[5],
            };
        }
        measurements
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct ReportMeasurement {
        raw: Measurement,
        estimated_total_ns: f64,
    }

    impl ReportMeasurement {
        fn add(&mut self, measurement: Measurement) {
            self.raw.calls += measurement.calls;
            self.raw.timed_calls += measurement.timed_calls;
            self.raw.sampled_ns += measurement.sampled_ns;
            self.raw.size_sum += measurement.size_sum;
            self.raw.max_size = self.raw.max_size.max(measurement.max_size);
            self.raw.zero_size_calls += measurement.zero_size_calls;
            // Weight each rank by its actual call count, not its sample count.
            self.estimated_total_ns += measurement.estimated_ns();
        }
    }

    fn write_csv<W: Write>(
        writer: &mut W,
        rank: &str,
        measurements: &[ReportMeasurement; Operation::ALL.len()],
        sample_interval: u64,
    ) -> io::Result<()> {
        writeln!(writer, "rank,operation,inclusive,calls,timed_calls,sampled_ns,estimated_total_ns,mean_ns,size_sum,mean_size,max_size,zero_size_calls,sample_interval")?;
        for (operation, measurement) in Operation::ALL.iter().zip(measurements) {
            let raw = measurement.raw;
            let mean_ns = (raw.calls > 0)
                .then(|| format!("{:.3}", measurement.estimated_total_ns / raw.calls as f64))
                .unwrap_or_default();
            let mean_size = (raw.calls > 0)
                .then(|| format!("{:.3}", raw.size_sum as f64 / raw.calls as f64))
                .unwrap_or_default();
            let inclusive = matches!(
                operation,
                Operation::RegistryInsert
                    | Operation::RegistryInsertWith
                    | Operation::DominanceScan
            );
            writeln!(
                writer,
                "{},{},{},{},{},{},{:.3},{},{},{},{},{},{}",
                rank,
                operation.name(),
                inclusive,
                raw.calls,
                raw.timed_calls,
                raw.sampled_ns,
                measurement.estimated_total_ns,
                mean_ns,
                raw.size_sum,
                mean_size,
                raw.max_size,
                raw.zero_size_calls,
                sample_interval
            )?;
        }
        Ok(())
    }

    /// Stop local recording and collectively gather onto rank zero. Rank zero
    /// writes every per-rank CSV and the aggregate; no shared-filesystem races.
    /// All ranks must call this after search with the same sample interval.
    pub fn finish_and_dump<C: Communicator>(
        communicator: &C,
        sample_interval: u64,
    ) -> io::Result<()> {
        let local = pack(&operation_timing::finish());
        let root = communicator.process_at_rank(0);
        if communicator.rank() != 0 {
            root.gather_into(&local[..]);
            return Ok(());
        }
        let mut received = vec![0u64; PACKED_LEN * communicator.size() as usize];
        root.gather_into_root(&local[..], &mut received[..]);
        let mut aggregate = [ReportMeasurement::default(); Operation::ALL.len()];
        for (rank, buffer) in received.chunks_exact(PACKED_LEN).enumerate() {
            let measurements = unpack(buffer);
            let mut report = [ReportMeasurement::default(); Operation::ALL.len()];
            for ((local, total), measurement) in
                report.iter_mut().zip(&mut aggregate).zip(measurements)
            {
                local.add(measurement);
                total.add(measurement);
            }
            let mut file =
                BufWriter::new(File::create(format!("operation_timing_rank_{rank}.csv"))?);
            write_csv(&mut file, &rank.to_string(), &report, sample_interval)?;
            file.flush()?;
        }
        let mut file = BufWriter::new(File::create("operation_timing.csv")?);
        write_csv(&mut file, "all", &aggregate, sample_interval)?;
        file.flush()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn packing_round_trip() {
            let mut measurements = [Measurement::default(); Operation::ALL.len()];
            measurements[2] = Measurement {
                calls: 9,
                timed_calls: 3,
                sampled_ns: 70,
                size_sum: 80,
                max_size: 33,
                zero_size_calls: 2,
            };
            assert_eq!(unpack(&pack(&measurements)), measurements);
        }

        #[test]
        fn aggregate_is_call_weighted_and_max_is_not_summed() {
            let mut total = ReportMeasurement::default();
            total.add(Measurement {
                calls: 10,
                timed_calls: 2,
                sampled_ns: 20,
                max_size: 30,
                ..Default::default()
            });
            total.add(Measurement {
                calls: 1,
                timed_calls: 1,
                sampled_ns: 100,
                max_size: 20,
                ..Default::default()
            });
            assert_eq!(total.raw.calls, 11);
            assert_eq!(total.raw.timed_calls, 3);
            assert_eq!(total.raw.sampled_ns, 120);
            assert_eq!(total.estimated_total_ns, 200.0);
            assert_eq!(total.raw.max_size, 30);
        }

        #[test]
        fn csv_includes_zero_call_operations_without_nan() {
            let report = [ReportMeasurement::default(); Operation::ALL.len()];
            let mut bytes = vec![];
            write_csv(&mut bytes, "all", &report, 1).unwrap();
            let csv = String::from_utf8(bytes).unwrap();
            assert_eq!(csv.lines().count(), Operation::ALL.len() + 1);
            assert!(csv.contains("all,heap_primary_push,false,0,0,0,0.000,,0,,0,0,1"));
            assert!(!csv.contains("NaN"));
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
    fn default_disabled() {
        let parameters = load("{}");
        assert!(!parameters.enabled);
        assert_eq!(parameters.sample_interval, 1);
        parameters.validate();
    }

    #[test]
    fn loads_sampling() {
        assert_eq!(
            load("operation_timing: true\noperation_timing_sample_interval: 100"),
            OperationTimingParameters {
                enabled: true,
                sample_interval: 100
            }
        );
    }

    #[test]
    fn rejects_invalid_intervals() {
        for value in ["0", "-1", "1.5", "true", "'10'"] {
            assert!(std::panic::catch_unwind(|| load(&format!(
                "operation_timing_sample_interval: {value}"
            )))
            .is_err());
        }
    }

    #[test]
    fn rejects_non_boolean_switch() {
        assert!(std::panic::catch_unwind(|| load("operation_timing: 1")).is_err());
    }

    #[cfg(not(feature = "operation-timing"))]
    #[test]
    #[should_panic(expected = "requires a build with --features operation-timing")]
    fn rejects_unavailable_instrumentation() {
        load("operation_timing: true").validate();
    }

    #[cfg(feature = "operation-timing")]
    #[test]
    fn accepts_available_instrumentation() {
        load("operation_timing: true").validate();
    }
}
