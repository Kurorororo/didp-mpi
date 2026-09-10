use mpi::Rank;
use std::fs::File;
use std::io::{self, BufWriter, Write};

/// A local event at which progressive-search width is observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WidthEvent {
    Start,
    LayerEnd,
    Goal,
    Refill,
    GoalRefill,
    Finish,
}

impl WidthEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::LayerEnd => "layer_end",
            Self::Goal => "goal",
            Self::Refill => "refill",
            Self::GoalRefill => "goal_refill",
            Self::Finish => "finish",
        }
    }
}

/// Snapshot immediately after a width update, before the next column/pack.
/// Queue lengths count heap entries, including entries awaiting lazy pruning.
#[derive(Clone, Debug, PartialEq)]
pub struct WidthStatisticsRecord {
    pub elapsed_time: f64,
    pub event: WidthEvent,
    pub previous_width: usize,
    pub width: usize,
    /// ACPS: the layer just visited; APPS: the last expanded node's depth.
    pub depth: Option<usize>,
    pub expanded: usize,
    pub generated: usize,
    pub sent: usize,
    pub received: usize,
    pub open_entries: usize,
    pub children_entries: usize,
    pub suspended_entries: usize,
}

/// Per-rank width history, recorded without MPI communication or search-time I/O.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WidthStatistics {
    records: Vec<WidthStatisticsRecord>,
}

impl WidthStatistics {
    /// Retains events even when a width cap or reset leaves the width unchanged.
    pub fn record(&mut self, record: WidthStatisticsRecord) {
        self.records.push(record);
    }

    pub fn iter(&self) -> impl Iterator<Item = &WidthStatisticsRecord> {
        self.records.iter()
    }

    pub fn dump_to_csv(&self, filename: &str, rank: Rank) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(filename)?);
        self.write_csv(&mut writer, rank)?;
        writer.flush()
    }

    fn write_csv<W: Write>(&self, mut writer: W, rank: Rank) -> io::Result<()> {
        writeln!(writer, "rank,elapsed_time,event,previous_width,width,depth,expanded,generated,sent,received,open_entries,children_entries,suspended_entries")?;
        for r in &self.records {
            let depth = r.depth.map_or_else(String::new, |d| d.to_string());
            writeln!(
                writer,
                "{rank},{},{},{},{},{depth},{},{},{},{},{},{},{}",
                r.elapsed_time,
                r.event.as_str(),
                r.previous_width,
                r.width,
                r.expanded,
                r.generated,
                r.sent,
                r.received,
                r.open_entries,
                r.children_entries,
                r.suspended_entries
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_resets_and_capped_width_events_in_csv() {
        let mut statistics = WidthStatistics::default();
        let mut record = WidthStatisticsRecord {
            elapsed_time: 1.25,
            event: WidthEvent::GoalRefill,
            previous_width: 7,
            width: 2,
            depth: Some(231),
            expanded: 100,
            generated: 150,
            sent: 40,
            received: 20,
            open_entries: 0,
            children_entries: 0,
            suspended_entries: 9,
        };
        statistics.record(record.clone());
        record.elapsed_time = 2.5;
        record.event = WidthEvent::Refill;
        record.previous_width = 2;
        record.expanded = 102;
        statistics.record(record);
        assert_eq!(statistics.iter().count(), 2);
        let mut csv = Vec::new();
        statistics.write_csv(&mut csv, 3).unwrap();
        assert_eq!(String::from_utf8(csv).unwrap(), concat!(
            "rank,elapsed_time,event,previous_width,width,depth,expanded,generated,sent,received,open_entries,children_entries,suspended_entries\n",
            "3,1.25,goal_refill,7,2,231,100,150,40,20,0,0,9\n",
            "3,2.5,refill,2,2,231,102,150,40,20,0,0,9\n",
        ));
    }

    #[test]
    fn rank_without_expansions_has_no_depth() {
        let mut statistics = WidthStatistics::default();
        statistics.record(WidthStatisticsRecord {
            elapsed_time: 0.0,
            event: WidthEvent::Start,
            previous_width: 1,
            width: 1,
            depth: None,
            expanded: 0,
            generated: 0,
            sent: 0,
            received: 0,
            open_entries: 0,
            children_entries: 0,
            suspended_entries: 0,
        });
        let mut csv = Vec::new();
        statistics.write_csv(&mut csv, 7).unwrap();
        assert!(String::from_utf8(csv)
            .unwrap()
            .ends_with("7,0,start,1,1,,0,0,0,0,0,0,0\n"));
    }
}
