use std::{
    error::Error,
    fs::OpenOptions,
    io::Write,
    ops::{Add, AddAssign},
};

use mpi::{collective::SystemOperation, traits::*, Rank, Tag};
use serde::Serialize;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Statistics {
    pub expanded: usize,
    pub generated: usize,
    pub sent: usize,
    pub kept: usize,
    pub received: usize,
    pub dominated_before_closed: usize,
    pub dominated_after_closed: usize,
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
    }
}

impl Statistics {
    pub fn dump_to_csv(list: &[Self], filename: &str) -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filename)?;

        let line = "rank,expanded,generated,sent,kept,received,dominated_before_closed,dominated_after_closed\n";
        file.write_all(line.as_bytes())?;

        for (rank, statistics) in list.iter().enumerate() {
            let line = format!(
                "{},{},{},{},{},{},{},{}\n",
                rank,
                statistics.expanded,
                statistics.generated,
                statistics.sent,
                statistics.kept,
                statistics.received,
                statistics.dominated_before_closed,
                statistics.dominated_after_closed
            );
            file.write_all(line.as_bytes())?;
        }

        Ok(())
    }

    pub fn send<C: Communicator>(&self, communicator: &C, destination_rank: Rank, tag: Tag) {
        let buffer = [
            self.expanded,
            self.generated,
            self.sent,
            self.kept,
            self.received,
            self.dominated_before_closed,
            self.dominated_after_closed,
        ];

        let destination = communicator.process_at_rank(destination_rank);
        destination.send_with_tag(&buffer[..], tag);
    }

    pub fn receive<C: Communicator>(communicator: &C, source_rank: Rank, tag: Tag) -> Self {
        let source = communicator.process_at_rank(source_rank);
        let mut buffer = [0; 7];
        source.receive_into_with_tag(&mut buffer[..], tag);

        Self {
            expanded: buffer[0],
            generated: buffer[1],
            sent: buffer[2],
            kept: buffer[3],
            received: buffer[4],
            dominated_before_closed: buffer[5],
            dominated_after_closed: buffer[6],
        }
    }

    pub fn reduce_sum<C: Communicator>(
        &self,
        communicator: &C,
        root_rank: Rank,
        is_root: bool,
    ) -> Option<Self> {
        let sendbuf = [
            self.expanded,
            self.generated,
            self.sent,
            self.kept,
            self.received,
            self.dominated_before_closed,
            self.dominated_after_closed,
        ];

        if is_root {
            let mut recvbuf = [0; 7];
            let root_process = communicator.process_at_rank(root_rank);
            root_process.reduce_into_root(&sendbuf, &mut recvbuf[..], SystemOperation::sum());
            Some(Self {
                expanded: recvbuf[0],
                generated: recvbuf[1],
                sent: recvbuf[2],
                kept: recvbuf[3],
                received: recvbuf[4],
                dominated_before_closed: recvbuf[5],
                dominated_after_closed: recvbuf[6],
            })
        } else {
            let root_process = communicator.process_at_rank(root_rank);
            root_process.reduce_into(&sendbuf, SystemOperation::sum());
            None
        }
    }

    pub fn gather<C: Communicator>(
        &self,
        communicator: &C,
        root_rank: Rank,
        is_root: bool,
    ) -> Vec<Statistics> {
        let sendbuf = [
            self.expanded,
            self.generated,
            self.sent,
            self.kept,
            self.received,
            self.dominated_before_closed,
            self.dominated_after_closed,
        ];

        if is_root {
            let n_ranks = communicator.size() as usize;
            let mut recvbuf = vec![0; 7 * n_ranks];
            let root_process = communicator.process_at_rank(root_rank);
            root_process.gather_into_root(&sendbuf, &mut recvbuf[..]);

            (0..n_ranks)
                .map(|rank| {
                    let offset = 7 * rank;
                    Self {
                        expanded: recvbuf[offset],
                        generated: recvbuf[offset + 1],
                        sent: recvbuf[offset + 2],
                        kept: recvbuf[offset + 3],
                        received: recvbuf[offset + 4],
                        dominated_before_closed: recvbuf[offset + 5],
                        dominated_after_closed: recvbuf[offset + 6],
                    }
                })
                .collect()
        } else {
            let root_process = communicator.process_at_rank(root_rank);
            root_process.gather_into(&sendbuf);
            vec![]
        }
    }
}
