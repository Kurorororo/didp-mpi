use mpi::{
    datatype::{DatatypeRef, MutView, UserDatatype, View},
    traits::*,
    Address, Count, Rank, Tag,
};
use std::mem;
use zerocopy::{AsBytes, FromBytes};

use crate::mpi_termination_detector::MpiTerminationDetector;

pub struct TimestampedCommunicator<'a, C> {
    communicator: &'a C,
    termination_detector: MpiTerminationDetector<'a, C>,
    offset: usize,
    datatype: UserDatatype,
    tag: Tag,
}

impl<'a, C> TimestampedCommunicator<'a, C>
where
    C: Communicator,
{
    pub fn new(
        communicator: &'a C,
        tag: Tag,
        tag_termination_detection: Tag,
        blocklengths: &[Count],
        displacements: &[Address],
        types: &[DatatypeRef<'static>],
        offset: Address,
    ) -> Self {
        let mut blocklengths = blocklengths.to_vec();
        blocklengths.push(1);
        let mut displacements = displacements.to_vec();
        displacements.push(offset);
        let mut types = types.to_vec();
        types.push(usize::equivalent_datatype());

        let termination_detector =
            MpiTerminationDetector::new(communicator, tag_termination_detection);
        let datatype = UserDatatype::structured(&blocklengths, &displacements, &types);

        Self {
            communicator,
            termination_detector,
            offset: offset as usize,
            datatype,
            tag,
        }
    }

    pub fn send(&mut self, buffer: &mut Vec<u8>, destination: Rank) {
        let tstamp = self.termination_detector.get_clock_to_send();
        let total_size = self.offset + mem::size_of::<usize>();
        buffer.resize(total_size, 0);
        buffer[self.offset..total_size].copy_from_slice(tstamp.as_bytes());
        let v = unsafe { View::with_count_and_datatype(&buffer[..], 1, &self.datatype) };
        let destination_process = self.communicator.process_at_rank(destination);
        destination_process.buffered_send_with_tag(&v, self.tag);
    }

    pub fn receive_into(&mut self, buffer: &mut Vec<u8>, source: Rank) {
        let total_size = self.offset + mem::size_of::<usize>();
        buffer.resize(total_size, 0);
        let mut v = unsafe { MutView::with_count_and_datatype(&mut buffer[..], 1, &self.datatype) };
        let source_process = self.communicator.process_at_rank(source);
        source_process.receive_into_with_tag(&mut v, self.tag);

        let tstamp = usize::read_from(&buffer[self.offset..total_size]).unwrap();
        self.termination_detector.notify_received(tstamp);

        buffer.truncate(self.offset);
    }

    pub fn initiate_termination(&mut self, destination_rank: Rank) {
        self.termination_detector.initiate(destination_rank);
    }

    pub fn receive_termination_detection_and_forward(
        &mut self,
        source_rank: Rank,
        destination_rank: Rank,
        local_invalid: bool,
    ) -> Option<bool> {
        self.termination_detector
            .receive_and_forward(source_rank, destination_rank, local_invalid)
    }
}
