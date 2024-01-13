use memoffset::offset_of;
use mpi::datatype::UserDatatype;
use mpi::{traits::*, Address, Rank, Tag};
use std::cmp::max;

pub struct MpiTerminationDetector<'a, C> {
    communicator: &'a C,
    previous: Rank,
    next: Rank,
    tag: Tag,
    clock: usize,
    tmax: usize,
    count: i32,
}

#[derive(Default)]
struct TerminationDetectionkMessage(usize, i32, bool, Rank);

unsafe impl Equivalence for TerminationDetectionkMessage {
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1, 1],
            &[
                offset_of!(TerminationDetectionkMessage, 0) as Address,
                offset_of!(TerminationDetectionkMessage, 1) as Address,
                offset_of!(TerminationDetectionkMessage, 2) as Address,
                offset_of!(TerminationDetectionkMessage, 3) as Address,
            ],
            &[
                usize::equivalent_datatype(),
                i32::equivalent_datatype(),
                bool::equivalent_datatype(),
                Rank::equivalent_datatype(),
            ],
        )
    }
}

impl<'a, C> MpiTerminationDetector<'a, C>
where
    C: Communicator,
{
    pub fn new(communicator: &'a C, tag: Tag) -> Self {
        let rank = communicator.rank();
        let size = communicator.size();
        let previous = (rank + size - 1) % size;
        let next = (rank + 1) % size;

        Self {
            communicator,
            tag,
            previous,
            next,
            clock: 0,
            tmax: 0,
            count: 0,
        }
    }

    pub fn get_clock_to_send(&mut self) -> usize {
        self.count += 1;
        self.clock
    }

    pub fn notify_received(&mut self, tstamp: usize) {
        self.clock = max(tstamp, self.clock);
        self.count -= 1;
    }

    pub fn initiate(&mut self) {
        self.clock += 1;
        let destination = self.communicator.process_at_rank(self.next);
        let message =
            TerminationDetectionkMessage(self.clock, self.count, false, self.communicator.rank());
        destination.buffered_send_with_tag(&message, self.tag);
    }

    pub fn check_and_forward(&mut self, local_invalid: bool) -> Option<bool> {
        let source = self.communicator.process_at_rank(self.previous);

        if source.immediate_probe_with_tag(self.tag).is_some() {
            let mut message = TerminationDetectionkMessage::default();
            source.receive_into_with_tag(&mut message, self.tag);

            self.clock = max(message.0, self.clock);
            let invalid = message.2 || local_invalid;

            if self.communicator.rank() == message.3 {
                Some(message.1 == 0 && !invalid)
            } else {
                let destination = self.communicator.process_at_rank(self.next);
                message.1 += self.count;
                message.2 = invalid || self.tmax >= message.0;

                destination.buffered_send_with_tag(&message, self.tag);

                Some(false)
            }
        } else {
            None
        }
    }
}
