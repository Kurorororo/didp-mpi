use std::ops::{Add, AddAssign};

use mpi::collective::SystemOperation;
use mpi::traits::*;
use mpi::{Rank, Tag};
use serde::Serialize;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Statistics {
    pub expanded: usize,
    pub generated: usize,
    pub sent: usize,
    pub kept: usize,
    pub received: usize,
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
    }
}

impl Statistics {
    pub fn send<C: Communicator>(&self, communicator: &C, destination_rank: Rank, tag: Tag) {
        let buffer = [
            self.expanded,
            self.generated,
            self.sent,
            self.kept,
            self.received,
        ];

        let destination = communicator.process_at_rank(destination_rank);
        destination.send_with_tag(&buffer[..], tag);
    }

    pub fn receive<C: Communicator>(communicator: &C, source_rank: Rank, tag: Tag) -> Self {
        let source = communicator.process_at_rank(source_rank);
        let mut buffer = [0; 5];
        source.receive_into_with_tag(&mut buffer[..], tag);

        Self {
            expanded: buffer[0],
            generated: buffer[1],
            sent: buffer[2],
            kept: buffer[3],
            received: buffer[4],
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
        ];

        if is_root {
            let mut recvbuf = [0; 5];
            communicator.process_at_rank(root_rank).reduce_into_root(
                &sendbuf,
                &mut recvbuf[..],
                SystemOperation::sum(),
            );
            Some(Self {
                expanded: recvbuf[0],
                generated: recvbuf[1],
                sent: recvbuf[2],
                kept: recvbuf[3],
                received: recvbuf[4],
            })
        } else {
            communicator
                .process_at_rank(root_rank)
                .reduce_into(&sendbuf, SystemOperation::sum());
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
        ];

        if is_root {
            let n_ranks = communicator.size() as usize;
            let mut recvbuf = vec![0; 5 * n_ranks];
            communicator
                .process_at_rank(root_rank)
                .gather_into_root(&sendbuf, &mut recvbuf[..]);

            (0..n_ranks)
                .map(|rank| {
                    let offset = 5 * rank;
                    Self {
                        expanded: recvbuf[offset],
                        generated: recvbuf[offset + 1],
                        sent: recvbuf[offset + 2],
                        kept: recvbuf[offset + 3],
                        received: recvbuf[offset + 4],
                    }
                })
                .collect()
        } else {
            communicator
                .process_at_rank(root_rank)
                .gather_into(&sendbuf);
            vec![]
        }
    }
}
