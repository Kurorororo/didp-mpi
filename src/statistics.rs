use std::{
    error::Error,
    fs::OpenOptions,
    io::Write,
    ops::{Add, AddAssign},
};

use mpi::{traits::*, Rank, Tag};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Statistics {
    pub expanded: usize,
    pub generated: usize,
    pub sent: usize,
    pub kept: usize,
    pub received: usize,
    pub dominated_before_closed: usize,
    pub dominated_after_closed: usize,
    pub first_expanded_timestamp: f64,
    pub last_expanded_timestamp: f64,
    pub first_received_timestamp: f64,
    pub last_received_timestamp: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatisticsTags {
    pub usize_tag: Tag,
    pub f64_tag: Tag,
}

impl Add for Statistics {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            expanded: self.expanded + rhs.expanded,
            generated: self.generated + rhs.generated,
            sent: self.sent + rhs.sent,
            kept: self.kept + rhs.kept,
            received: self.received + rhs.received,
            dominated_before_closed: self.dominated_before_closed + rhs.dominated_before_closed,
            dominated_after_closed: self.dominated_after_closed + rhs.dominated_after_closed,
            first_expanded_timestamp: if self.first_expanded_timestamp
                < rhs.first_expanded_timestamp
            {
                self.first_expanded_timestamp
            } else {
                rhs.first_expanded_timestamp
            },
            last_expanded_timestamp: if self.last_expanded_timestamp > rhs.last_expanded_timestamp {
                self.last_expanded_timestamp
            } else {
                rhs.last_expanded_timestamp
            },
            first_received_timestamp: if self.first_received_timestamp
                < rhs.first_received_timestamp
            {
                self.first_received_timestamp
            } else {
                rhs.first_received_timestamp
            },
            last_received_timestamp: if self.last_received_timestamp > rhs.last_received_timestamp {
                self.last_received_timestamp
            } else {
                rhs.last_received_timestamp
            },
        }
    }
}

impl AddAssign for Statistics {
    fn add_assign(&mut self, rhs: Self) {
        self.expanded += rhs.expanded;
        self.generated += rhs.generated;
        self.sent += rhs.sent;
        self.kept += rhs.kept;
        self.received += rhs.received;
        self.dominated_before_closed += rhs.dominated_before_closed;
        self.dominated_after_closed += rhs.dominated_after_closed;

        if self.first_expanded_timestamp > rhs.first_expanded_timestamp {
            self.first_expanded_timestamp = rhs.first_expanded_timestamp;
        }

        if self.last_expanded_timestamp < rhs.last_expanded_timestamp {
            self.last_expanded_timestamp = rhs.last_expanded_timestamp;
        }

        if self.first_received_timestamp > rhs.first_received_timestamp {
            self.first_received_timestamp = rhs.first_received_timestamp;
        }

        if self.last_received_timestamp < rhs.last_received_timestamp {
            self.last_received_timestamp = rhs.last_received_timestamp;
        }
    }
}

impl Statistics {
    pub fn dump_to_csv(list: &[Self], filename: &str) -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filename)?;

        let line = "rank,expanded,generated,sent,kept,received,dominated_before_closed,dominated_after_closed,first_expanded_timestamp,last_expanded_timestamp,first_received_timestamp,last_received_timestamp\n";
        file.write_all(line.as_bytes())?;

        for (rank, statistics) in list.iter().enumerate() {
            let line = format!(
                "{},{},{},{},{},{},{},{},{},{},{},{}\n",
                rank,
                statistics.expanded,
                statistics.generated,
                statistics.sent,
                statistics.kept,
                statistics.received,
                statistics.dominated_before_closed,
                statistics.dominated_after_closed,
                statistics.first_expanded_timestamp,
                statistics.last_expanded_timestamp,
                statistics.first_received_timestamp,
                statistics.last_received_timestamp
            );
            file.write_all(line.as_bytes())?;
        }

        Ok(())
    }

    pub fn send<C: Communicator>(
        &self,
        communicator: &C,
        destination_rank: Rank,
        tags: &StatisticsTags,
    ) {
        let destination = communicator.process_at_rank(destination_rank);

        let usize_buffer = [
            self.expanded,
            self.generated,
            self.sent,
            self.kept,
            self.received,
            self.dominated_before_closed,
            self.dominated_after_closed,
        ];

        destination.send_with_tag(&usize_buffer[..], tags.usize_tag);

        let f64_buffer = [
            self.first_expanded_timestamp,
            self.last_expanded_timestamp,
            self.first_received_timestamp,
            self.last_received_timestamp,
        ];

        destination.send_with_tag(&f64_buffer[..], tags.f64_tag);
    }

    pub fn receive<C: Communicator>(
        communicator: &C,
        source_rank: Rank,
        tags: &StatisticsTags,
    ) -> Self {
        let source = communicator.process_at_rank(source_rank);

        let mut usize_buffer = [0; 7];
        source.receive_into_with_tag(&mut usize_buffer[..], tags.usize_tag);

        let mut f64_buffer = [0.0; 4];
        source.receive_into_with_tag(&mut f64_buffer[..], tags.f64_tag);

        Self {
            expanded: usize_buffer[0],
            generated: usize_buffer[1],
            sent: usize_buffer[2],
            kept: usize_buffer[3],
            received: usize_buffer[4],
            dominated_before_closed: usize_buffer[5],
            dominated_after_closed: usize_buffer[6],
            first_expanded_timestamp: f64_buffer[0],
            last_expanded_timestamp: f64_buffer[1],
            first_received_timestamp: f64_buffer[2],
            last_received_timestamp: f64_buffer[3],
        }
    }

    pub fn gather<C: Communicator>(
        &self,
        communicator: &C,
        root_rank: Rank,
        is_root: bool,
    ) -> Vec<Statistics> {
        let usize_sendbuf = [
            self.expanded,
            self.generated,
            self.sent,
            self.kept,
            self.received,
            self.dominated_before_closed,
            self.dominated_after_closed,
        ];
        let f64_sendbuf = [
            self.first_expanded_timestamp,
            self.last_expanded_timestamp,
            self.first_received_timestamp,
            self.last_received_timestamp,
        ];

        if is_root {
            let n_ranks = communicator.size() as usize;
            let root_process = communicator.process_at_rank(root_rank);

            let mut usize_recvbuf = vec![0; 7 * n_ranks];
            root_process.gather_into_root(&usize_sendbuf, &mut usize_recvbuf[..]);

            let mut f64_recvbuf = vec![0.0; 4 * n_ranks];
            root_process.gather_into_root(&f64_sendbuf, &mut f64_recvbuf[..]);

            (0..n_ranks)
                .map(|rank| {
                    let usize_offset = 7 * rank;
                    let f64_offset = 4 * rank;

                    Self {
                        expanded: usize_recvbuf[usize_offset],
                        generated: usize_recvbuf[usize_offset + 1],
                        sent: usize_recvbuf[usize_offset + 2],
                        kept: usize_recvbuf[usize_offset + 3],
                        received: usize_recvbuf[usize_offset + 4],
                        dominated_before_closed: usize_recvbuf[usize_offset + 5],
                        dominated_after_closed: usize_recvbuf[usize_offset + 6],
                        first_expanded_timestamp: f64_recvbuf[f64_offset],
                        last_expanded_timestamp: f64_recvbuf[f64_offset + 1],
                        first_received_timestamp: f64_recvbuf[f64_offset + 2],
                        last_received_timestamp: f64_recvbuf[f64_offset + 3],
                    }
                })
                .collect()
        } else {
            let root_process = communicator.process_at_rank(root_rank);
            root_process.gather_into(&usize_sendbuf);
            root_process.gather_into(&f64_sendbuf);
            vec![]
        }
    }
}
