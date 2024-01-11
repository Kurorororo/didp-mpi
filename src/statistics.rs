use mpi::collective::SystemOperation;
use mpi::traits::*;
use mpi::{Rank, Tag};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Statistics {
    pub expanded: usize,
    pub generated: usize,
    pub sent: usize,
    pub kept: usize,
    pub received: usize,
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
}
