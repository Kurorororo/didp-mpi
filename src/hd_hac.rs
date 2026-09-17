use didp_yaml::heuristic_search_solver::CostToDump;
use dypdl::{prelude::*, variable_type::Numeric};
#[cfg(feature = "operation-timing")]
use dypdl_heuristic_search::operation_timing::{Operation, Timer};
use dypdl_heuristic_search::search_algorithm::{
    data_structure::{HashableSignatureVariables, StateWithHashableSignatureVariables},
    SearchInput, Solution, StateInRegistry, StateRegistry, TransitionWithId,
};
use mpi::{topology::SimpleCommunicator, traits::*, Rank, Tag};
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::{self, Write};
use std::mem;
use std::rc::Rc;
use std::str::FromStr;

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;
use crate::node_communicator::TimeStampedNodeDepthCommunicator;
#[cfg(feature = "memory-statistics")]
use crate::node_memory::LiveNodeCounter;
use crate::open_list;
use crate::{bfs_node_with_distributed_id_chain::NodeGenerationResult, is_float::IsFloat};
use crate::{
    distributed_id_chain::DistributedTransitionIdChain,
    mpi_anytime_search::{
        MpiAnytimeSearch, MpiAnytimeSearchEvaluators, MpiAnytimeSearchParameters, TAG_OFFSET,
    },
};
use crate::{node_message::NodeMessage, statistics::Statistics, ExpansionStatistics};

const TAG_NODE: Tag = TAG_OFFSET;
const TAG_TIME_OUT: Tag = TAG_OFFSET + 1;
const TAG_TIME_OUT_ACK: Tag = TAG_OFFSET + 2;
const TAG_TERMINATION_DETECTION: Tag = TAG_OFFSET + 3;
const TAG_TERMINATE: Tag = TAG_OFFSET + 4;

#[derive(Clone, Debug, PartialEq)]
pub struct HdHacMemoryStatistics {
    pub rank: Rank,
    pub elapsed_time: f64,
    pub resident_memory_bytes: Option<u64>,
    pub peak_resident_memory_bytes: Option<u64>,
    pub virtual_memory_bytes: Option<u64>,
    pub estimated_node_and_state_bytes_per_search_node: usize,
    pub estimated_retained_node_and_state_bytes: usize,
    pub open_list_allocated_bytes: usize,
    pub transition_chain_allocated_bytes: usize,
    pub estimated_search_data_structure_bytes: usize,
    pub estimated_search_data_structure_bytes_per_search_node: Option<f64>,
    pub primary_open_len: usize,
    pub primary_open_capacity: usize,
    pub layered_open_len: usize,
    pub layered_open_capacity: usize,
    pub layered_open_layers: usize,
    pub registry_entries: usize,
    pub registry_open_entries: usize,
    pub registry_closed_entries: usize,
    pub registry_inserted: usize,
    pub registry_removed: usize,
    pub transition_chain_nodes: usize,
    pub expanded: usize,
    pub generated: usize,
    pub kept: usize,
    pub sent: usize,
    pub received: usize,
    pub memory_estimate_version: usize,
    pub live_nodes: usize,
    pub live_nodes_created: usize,
    pub live_nodes_dropped: usize,
    pub live_nodes_outside_registry: usize,
    pub registry_signatures: usize,
    pub estimated_private_node_and_resource_bytes: usize,
    pub estimated_shared_signature_bytes: usize,
    pub estimated_registry_storage_bytes: usize,
}

impl HdHacMemoryStatistics {
    pub fn dump_to_csv(list: &[Self], filename: &str) -> Result<(), Box<dyn Error>> {
        let mut file = Self::create_csv(filename)?;

        for statistics in list {
            statistics.write_csv_row(&mut file)?;
        }

        Ok(())
    }

    fn create_csv(filename: &str) -> io::Result<File> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filename)?;
        file.write_all(
            b"rank,elapsed_time,resident_memory_bytes,peak_resident_memory_bytes,virtual_memory_bytes,estimated_node_and_state_bytes_per_search_node,estimated_retained_node_and_state_bytes,open_list_allocated_bytes,transition_chain_allocated_bytes,estimated_search_data_structure_bytes,estimated_search_data_structure_bytes_per_search_node,primary_open_len,primary_open_capacity,layered_open_len,layered_open_capacity,layered_open_layers,registry_entries,registry_open_entries,registry_closed_entries,registry_inserted,registry_removed,transition_chain_nodes,expanded,generated,kept,sent,received,",
        )?;
        file.write_all(b"memory_estimate_version,live_nodes,live_nodes_created,live_nodes_dropped,live_nodes_outside_registry,registry_signatures,estimated_private_node_and_resource_bytes,estimated_shared_signature_bytes,estimated_registry_storage_bytes\n")?;

        Ok(file)
    }

    fn write_csv_row(&self, file: &mut File) -> io::Result<()> {
        let line = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},",
            self.rank,
            self.elapsed_time,
            self.resident_memory_bytes
                .map_or_else(String::new, |value| value.to_string()),
            self.peak_resident_memory_bytes
                .map_or_else(String::new, |value| value.to_string()),
            self.virtual_memory_bytes
                .map_or_else(String::new, |value| value.to_string()),
            self.estimated_node_and_state_bytes_per_search_node,
            self.estimated_retained_node_and_state_bytes,
            self.open_list_allocated_bytes,
            self.transition_chain_allocated_bytes,
            self.estimated_search_data_structure_bytes,
            self.estimated_search_data_structure_bytes_per_search_node
                .map_or_else(String::new, |value| value.to_string()),
            self.primary_open_len,
            self.primary_open_capacity,
            self.layered_open_len,
            self.layered_open_capacity,
            self.layered_open_layers,
            self.registry_entries,
            self.registry_open_entries,
            self.registry_closed_entries,
            self.registry_inserted,
            self.registry_removed,
            self.transition_chain_nodes,
            self.expanded,
            self.generated,
            self.kept,
            self.sent,
            self.received,
        );
        file.write_all(line.as_bytes())?;
        let components = format!(
            "{},{},{},{},{},{},{},{},{}\n",
            self.memory_estimate_version,
            self.live_nodes,
            self.live_nodes_created,
            self.live_nodes_dropped,
            self.live_nodes_outside_registry,
            self.registry_signatures,
            self.estimated_private_node_and_resource_bytes,
            self.estimated_shared_signature_bytes,
            self.estimated_registry_storage_bytes
        );
        file.write_all(components.as_bytes())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HdHacSearchNodeMemoryLayout {
    pub search_node_inline_bytes: usize,
    pub state_inline_bytes: usize,
    pub signature_variables_inline_bytes: usize,
    pub state_variable_payload_bytes: usize,
    pub state_registry_entry_bytes: usize,
    pub transition_chain_bytes: usize,
    pub primary_open_entry_bytes: usize,
    pub layered_open_entry_bytes: usize,
    pub estimated_node_and_state_bytes: usize,
    pub private_node_and_resource_bytes: usize,
    pub shared_signature_bytes: usize,
    pub registry_signature_entry_bytes: usize,
    pub registry_node_reference_bytes: usize,
}

impl HdHacSearchNodeMemoryLayout {
    fn with_model<N>(model: &Model) -> Self {
        // Rc currently stores one strong and one weak counter alongside the value. This is an
        // estimate because its allocation layout is not a stable Rust API.
        let rc_allocation_overhead = 2 * mem::size_of::<usize>();
        let search_node_inline_bytes = mem::size_of::<N>();
        let state_inline_bytes = mem::size_of::<StateInRegistry>();
        let signature_variables_inline_bytes = mem::size_of::<HashableSignatureVariables>();
        let state_variable_payload_bytes = state_variable_payload_bytes(&model.target);
        let registry_signature_entry_bytes =
            mem::size_of::<(Rc<HashableSignatureVariables>, Vec<Rc<N>>)>();
        let registry_node_reference_bytes = mem::size_of::<Rc<N>>();
        let state_registry_entry_bytes =
            registry_signature_entry_bytes + registry_node_reference_bytes;
        let resource_payload_bytes = state_resource_payload_bytes(&model.target);
        let private_node_and_resource_bytes =
            rc_allocation_overhead + search_node_inline_bytes + resource_payload_bytes;
        let shared_signature_bytes = rc_allocation_overhead
            + signature_variables_inline_bytes
            + state_variable_payload_bytes
            - resource_payload_bytes;
        let transition_chain_bytes =
            rc_allocation_overhead + mem::size_of::<DistributedTransitionIdChain>();
        let estimated_node_and_state_bytes = rc_allocation_overhead
            + search_node_inline_bytes
            + rc_allocation_overhead
            + signature_variables_inline_bytes
            + state_variable_payload_bytes
            + state_registry_entry_bytes;

        Self {
            search_node_inline_bytes,
            state_inline_bytes,
            signature_variables_inline_bytes,
            state_variable_payload_bytes,
            state_registry_entry_bytes,
            transition_chain_bytes,
            primary_open_entry_bytes: mem::size_of::<(Rc<N>, usize)>(),
            layered_open_entry_bytes: mem::size_of::<Rc<N>>(),
            estimated_node_and_state_bytes,
            private_node_and_resource_bytes,
            shared_signature_bytes,
            registry_signature_entry_bytes,
            registry_node_reference_bytes,
        }
    }

    #[cfg(feature = "memory-statistics")]
    fn retained_components(
        &self,
        live_nodes: usize,
        signatures: usize,
        registry_nodes: usize,
    ) -> (usize, usize, usize) {
        (
            live_nodes.saturating_mul(self.private_node_and_resource_bytes),
            signatures.saturating_mul(self.shared_signature_bytes),
            signatures
                .saturating_mul(self.registry_signature_entry_bytes)
                .saturating_add(registry_nodes.saturating_mul(self.registry_node_reference_bytes)),
        )
    }
}

#[cfg(any(test, feature = "memory-statistics"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ProcessMemory {
    resident_bytes: Option<u64>,
    peak_resident_bytes: Option<u64>,
    virtual_bytes: Option<u64>,
}

#[cfg(any(test, feature = "memory-statistics"))]
fn parse_linux_process_memory(status: &str) -> ProcessMemory {
    fn value_in_bytes(status: &str, key: &str) -> Option<u64> {
        let line = status.lines().find(|line| line.starts_with(key))?;
        let mut values = line.split_whitespace();
        values.next()?;
        let value = values.next()?.parse::<u64>().ok()?;
        match values.next() {
            Some("kB") => value.checked_mul(1024),
            None => Some(value),
            _ => None,
        }
    }

    ProcessMemory {
        resident_bytes: value_in_bytes(status, "VmRSS:"),
        peak_resident_bytes: value_in_bytes(status, "VmHWM:"),
        virtual_bytes: value_in_bytes(status, "VmSize:"),
    }
}

#[cfg(feature = "memory-statistics")]
fn process_memory() -> ProcessMemory {
    std::fs::read_to_string("/proc/self/status")
        .map(|status| parse_linux_process_memory(&status))
        .unwrap_or_default()
}

fn state_variable_payload_bytes<S: StateInterface>(state: &S) -> usize {
    let set_bytes = (0..state.get_number_of_set_variables())
        .map(|i| mem::size_of::<Set>() + mem::size_of_val(state.get_set_variable(i).as_slice()))
        .sum::<usize>();
    let vector_bytes = (0..state.get_number_of_vector_variables())
        .map(|i| {
            mem::size_of::<Vector>()
                + state.get_vector_variable(i).len() * mem::size_of::<Element>()
        })
        .sum::<usize>();

    set_bytes
        + vector_bytes
        + state.get_number_of_element_variables() * mem::size_of::<Element>()
        + state.get_number_of_integer_variables() * mem::size_of::<Integer>()
        + state.get_number_of_continuous_variables() * mem::size_of::<Continuous>()
        + state.get_number_of_element_resource_variables() * mem::size_of::<Element>()
        + state.get_number_of_integer_resource_variables() * mem::size_of::<Integer>()
        + state.get_number_of_continuous_resource_variables() * mem::size_of::<Continuous>()
}

fn state_resource_payload_bytes<S: StateInterface>(state: &S) -> usize {
    state.get_number_of_element_resource_variables() * mem::size_of::<Element>()
        + state.get_number_of_integer_resource_variables() * mem::size_of::<Integer>()
        + state.get_number_of_continuous_resource_variables() * mem::size_of::<Continuous>()
}

pub struct HdHac<'a, T, N, M, L, R, B, F, V = Transition>
where
    T: Numeric + IsFloat + Ord + Display,
    N: BfsNodeWithDistributedIdChain<T> + From<M>,
    V: TransitionInterface + Clone + Default,
{
    model: Rc<Model>,
    search: MpiAnytimeSearch<'a, T, N, M, L, R, B, F, V>,
    communicator: &'a SimpleCommunicator,
    node_communicator: TimeStampedNodeDepthCommunicator<'a, SimpleCommunicator, M, T>,
    open: BinaryHeap<(Rc<N>, usize)>,
    layered_open: Vec<BinaryHeap<Rc<N>>>,
    registry: StateRegistry<T, N>,
    current_depth: usize,
    is_layered_turn: bool,
    local_dual_bound: Option<T>,
    is_time_out: bool,
    n_remaining_time_out_ack: usize,
    is_checking_termination: bool,
    is_terminated: bool,
    memory_monitoring_interval: Option<f64>,
    #[cfg(feature = "memory-statistics")]
    next_memory_monitoring_time: f64,
    search_node_memory_layout: HdHacSearchNodeMemoryLayout,
    memory_statistics: Vec<HdHacMemoryStatistics>,
    #[cfg(feature = "memory-statistics")]
    memory_statistics_file: Option<File>,
    expansion_statistics: Option<ExpansionStatistics<T>>,
}

impl<'a, T, N, M, L, R, B, F, V> HdHac<'a, T, N, M, L, R, B, F, V>
where
    T: Numeric + IsFloat + Ord + Display + Hash,
    <T as FromStr>::Err: Debug,
    CostToDump: From<T>,
    N: BfsNodeWithDistributedIdChain<T> + From<M> + Clone,
    M: Clone + NodeMessage<T> + From<N>,
    L: FnMut(
        StateInRegistry,
        T,
        &TransitionWithId<V>,
        &DistributedTransitionIdChain,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> NodeGenerationResult<Rc<N>>,
    R: FnMut(
        StateWithHashableSignatureVariables,
        T,
        &TransitionWithId<V>,
        &DistributedTransitionIdChain,
        Option<T>,
    ) -> Option<M>,
    B: FnMut(T, T) -> T,
    F: FnMut(&HashableSignatureVariables) -> u64,
    V: TransitionInterface + Clone + Default + 'static,
    Transition: From<V> + From<TransitionWithId<V>>,
    TransitionWithId<V>: Clone,
{
    /// Creates a new HDACPS solver.
    pub fn new(
        input: SearchInput<'a, M, TransitionWithId<V>>,
        evaluators: MpiAnytimeSearchEvaluators<L, R, B>,
        parameters: MpiAnytimeSearchParameters<T>,
        hash_function: F,
        communicator: &'a SimpleCommunicator,
    ) -> Self {
        let model = input.generator.model.clone();
        let search_node_memory_layout = HdHacSearchNodeMemoryLayout::with_model::<N>(&model);
        let capacity = parameters.parameters.initial_registry_capacity;
        let mut search = MpiAnytimeSearch::new(
            input.generator,
            input.solution_suffix,
            evaluators,
            parameters,
            hash_function,
            communicator,
        );

        let node_communicator = TimeStampedNodeDepthCommunicator::new(
            communicator,
            TAG_NODE,
            TAG_TERMINATION_DETECTION,
            model.clone(),
        );

        let mut open = BinaryHeap::new();
        let layered_open = vec![BinaryHeap::new()];
        let mut registry = StateRegistry::new(model.clone());

        if let Some(capacity) = capacity {
            registry.reserve(capacity);
        }

        if let Some(node) = search.generate_root_node(input.node, &mut registry) {
            open_list::push_primary(&mut open, (node, 0));
        }

        Self {
            model,
            search,
            communicator,
            node_communicator,
            open,
            layered_open,
            registry,
            current_depth: 0,
            is_layered_turn: false,
            local_dual_bound: None,
            is_time_out: false,
            n_remaining_time_out_ack: 0,
            is_checking_termination: false,
            is_terminated: false,
            memory_monitoring_interval: None,
            #[cfg(feature = "memory-statistics")]
            next_memory_monitoring_time: 0.0,
            search_node_memory_layout,
            memory_statistics: vec![],
            #[cfg(feature = "memory-statistics")]
            memory_statistics_file: None,
            expansion_statistics: None,
        }
    }

    /// Enables memory monitoring, immediately writing the first sample and each subsequent
    /// sample to `memory_statistics_rank_<rank>.csv`. Call this before [`Self::search`].
    /// Requires the Cargo feature `memory-statistics`.
    #[cfg(feature = "memory-statistics")]
    pub fn enable_memory_monitoring(&mut self, interval: f64) {
        assert!(
            interval.is_finite() && interval > 0.0,
            "memory monitoring interval must be positive and finite"
        );
        assert!(
            self.memory_monitoring_interval.is_none(),
            "memory monitoring is already enabled"
        );
        let filename = format!("memory_statistics_rank_{}.csv", self.communicator.rank());
        self.memory_statistics_file = Some(
            HdHacMemoryStatistics::create_csv(&filename)
                .expect("failed to create memory statistics CSV"),
        );
        let counter = Rc::new(LiveNodeCounter::default());
        for (node, _) in &self.open {
            node.track_memory(&counter);
        }
        self.search
            .enable_state_registry_statistics(self.open.len(), counter);
        self.memory_monitoring_interval = Some(interval);
        self.next_memory_monitoring_time = self.search.elapsed_time();
        self.record_memory_statistics_if_due();
    }

    /// Memory monitoring requires a build with `--features memory-statistics`.
    #[cfg(not(feature = "memory-statistics"))]
    pub fn enable_memory_monitoring(&mut self, _interval: f64) {
        panic!("memory monitoring requires a build with --features memory-statistics");
    }

    pub fn memory_statistics(&self) -> &[HdHacMemoryStatistics] {
        &self.memory_statistics
    }

    pub fn enable_expansion_statistics(&mut self) {
        assert!(
            self.search.local_statistics().expanded == 0,
            "expansion statistics must be enabled before search"
        );
        self.expansion_statistics = Some(ExpansionStatistics::default());
    }

    pub fn expansion_statistics(&self) -> Option<&ExpansionStatistics<T>> {
        self.expansion_statistics.as_ref()
    }

    pub fn search_node_memory_layout(&self) -> HdHacSearchNodeMemoryLayout {
        self.search_node_memory_layout
    }

    #[cfg(not(feature = "memory-statistics"))]
    #[inline]
    fn record_memory_statistics_if_due(&mut self) {}

    #[cfg(feature = "memory-statistics")]
    fn record_memory_statistics_if_due(&mut self) {
        let Some(interval) = self.memory_monitoring_interval else {
            return;
        };
        let elapsed_time = self.search.elapsed_time();

        if elapsed_time < self.next_memory_monitoring_time {
            return;
        }

        let memory = process_memory();
        let local_statistics = self.search.local_statistics();
        let registry = self.search.state_registry_statistics();
        let layered_open_len = self.layered_open.iter().map(BinaryHeap::len).sum();
        let layered_open_capacity: usize = self.layered_open.iter().map(BinaryHeap::capacity).sum();
        let open_list_allocated_bytes = self
            .open
            .capacity()
            .saturating_mul(self.search_node_memory_layout.primary_open_entry_bytes)
            .saturating_add(
                layered_open_capacity
                    .saturating_mul(self.search_node_memory_layout.layered_open_entry_bytes),
            )
            .saturating_add(
                self.layered_open
                    .capacity()
                    .saturating_mul(mem::size_of::<BinaryHeap<Rc<N>>>()),
            );
        let transition_chain_allocated_bytes = self
            .search
            .transition_chain_nodes()
            .saturating_mul(self.search_node_memory_layout.transition_chain_bytes)
            .saturating_add(
                self.search
                    .transition_chain_capacity()
                    .saturating_mul(mem::size_of::<Rc<DistributedTransitionIdChain>>()),
            );
        let live_nodes = self.search.live_node_statistics();
        let registry_signatures = self.registry.signature_count();
        let (
            estimated_private_node_and_resource_bytes,
            estimated_shared_signature_bytes,
            estimated_registry_storage_bytes,
        ) = self.search_node_memory_layout.retained_components(
            live_nodes.live,
            registry_signatures,
            registry.entries,
        );
        let estimated_retained_node_and_state_bytes = estimated_private_node_and_resource_bytes
            .saturating_add(estimated_shared_signature_bytes)
            .saturating_add(estimated_registry_storage_bytes);
        let estimated_search_data_structure_bytes = estimated_retained_node_and_state_bytes
            .saturating_add(open_list_allocated_bytes)
            .saturating_add(transition_chain_allocated_bytes);
        let estimated_search_data_structure_bytes_per_search_node = (live_nodes.live > 0)
            .then(|| estimated_search_data_structure_bytes as f64 / live_nodes.live as f64);
        let statistics = HdHacMemoryStatistics {
            rank: self.communicator.rank(),
            elapsed_time,
            resident_memory_bytes: memory.resident_bytes,
            peak_resident_memory_bytes: memory.peak_resident_bytes,
            virtual_memory_bytes: memory.virtual_bytes,
            estimated_node_and_state_bytes_per_search_node: self
                .search_node_memory_layout
                .estimated_node_and_state_bytes,
            estimated_retained_node_and_state_bytes,
            open_list_allocated_bytes,
            transition_chain_allocated_bytes,
            estimated_search_data_structure_bytes,
            estimated_search_data_structure_bytes_per_search_node,
            primary_open_len: self.open.len(),
            primary_open_capacity: self.open.capacity(),
            layered_open_len,
            layered_open_capacity,
            layered_open_layers: self.layered_open.len(),
            registry_entries: registry.entries,
            registry_open_entries: registry.entries - registry.closed_entries,
            registry_closed_entries: registry.closed_entries,
            registry_inserted: registry.inserted,
            registry_removed: registry.removed,
            transition_chain_nodes: self.search.transition_chain_nodes(),
            expanded: local_statistics.expanded,
            generated: local_statistics.generated,
            kept: local_statistics.kept,
            sent: local_statistics.sent,
            received: local_statistics.received,
            memory_estimate_version: 2,
            live_nodes: live_nodes.live,
            live_nodes_created: live_nodes.created,
            live_nodes_dropped: live_nodes.dropped,
            live_nodes_outside_registry: live_nodes
                .live
                .checked_sub(registry.entries)
                .expect("registry entries exceed tracked live nodes"),
            registry_signatures,
            estimated_private_node_and_resource_bytes,
            estimated_shared_signature_bytes,
            estimated_registry_storage_bytes,
        };
        if let Some(file) = self.memory_statistics_file.as_mut() {
            statistics
                .write_csv_row(file)
                .expect("failed to write memory statistics CSV");
        }
        self.memory_statistics.push(statistics);

        while self.next_memory_monitoring_time <= elapsed_time {
            self.next_memory_monitoring_time += interval;
        }
    }

    pub fn set_time_offset(&mut self, offset: f64) {
        self.search.set_time_offset(offset);
    }

    pub fn close_root_node(&mut self, node: M) {
        self.search.close_root_node(node, &mut self.registry)
    }

    fn open_node(&mut self, node: N, depth: usize) {
        if let Some(node) = self.search.open_node(node, &mut self.registry) {
            open_list::push_primary(&mut self.open, (node.clone(), depth));

            while depth >= self.layered_open.len() {
                self.layered_open.push(BinaryHeap::new());
            }

            open_list::push_layered(&mut self.layered_open[depth], node);
        }
    }

    fn receive_node(&mut self, source_rank: Rank) {
        self.search.increment_received();

        if self.is_time_out {
            if let Some(bound) = self
                .node_communicator
                .receive_and_discard(source_rank, self.local_dual_bound)
            {
                if N::ordered_by_bound() {
                    self.local_dual_bound = Some(bound);
                }
            }
        } else if let Some((node, depth)) = self
            .node_communicator
            .receive(source_rank, self.search.get_primal_bound())
        {
            let node = N::from(node);
            self.open_node(node, depth);
        }
    }

    fn broadcast_time_out(&mut self) {
        for destination_rank in 0..self.communicator.size() {
            if destination_rank != self.communicator.rank() {
                let buffer: [u8; 0] = [];
                let destination_process = self.communicator.process_at_rank(destination_rank);
                destination_process.buffered_send_with_tag(&buffer, TAG_TIME_OUT);
                self.n_remaining_time_out_ack += 1;
            }
        }
    }

    fn receive_time_out(&mut self, source_rank: Rank) {
        let mut buffer: [u8; 0] = [];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, TAG_TIME_OUT);
        self.is_time_out = true;
        self.local_dual_bound = self.compute_local_dual_bound();
        source_process.buffered_send_with_tag(&buffer, TAG_TIME_OUT_ACK);
    }

    fn receive_time_out_ack(&mut self, source_rank: Rank) {
        let mut buffer: [u8; 0] = [];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, TAG_TIME_OUT_ACK);
        self.n_remaining_time_out_ack -= 1;
    }

    fn broadcast_terminate(&mut self) {
        for destination_rank in 0..self.communicator.size() {
            if destination_rank != self.communicator.rank() {
                let buffer: [u8; 0] = [];
                let destination_process = self.communicator.process_at_rank(destination_rank);
                destination_process.buffered_send_with_tag(&buffer, TAG_TERMINATE);
            }
        }

        self.is_terminated = true;
    }

    fn receive_terminate(&mut self, source_rank: Rank) {
        let mut buffer: [u8; 0] = [];
        let source_process = self.communicator.process_at_rank(source_rank);
        source_process.receive_into_with_tag(&mut buffer, TAG_TERMINATE);
        self.is_terminated = true;
    }

    fn receive_termination_detection(&mut self, source_rank: Rank) {
        let destination_rank = (self.communicator.rank() + 1) % self.communicator.size();
        let local_invalid = self.search.cannot_terminate()
            || self.n_remaining_time_out_ack > 0
            || (!self.is_time_out && !self.open.is_empty());
        let result = self
            .node_communicator
            .receive_termination_detection_and_forward(
                source_rank,
                destination_rank,
                local_invalid,
            );

        if let Some(result) = result {
            if result {
                self.broadcast_terminate();
            }

            self.is_checking_termination = false;
        }
    }

    fn process_message(&mut self) {
        let any_process = self.communicator.any_process();

        while let Some(status) = {
            #[cfg(feature = "operation-timing")]
            let _timer = Timer::start(Operation::MessageProbe, 1);
            any_process.immediate_probe()
        } {
            #[cfg(feature = "operation-timing")]
            let _timer = Timer::start(Operation::MessageDispatch, 1);
            let source_rank = status.source_rank();
            let tag = status.tag();

            match tag {
                TAG_NODE => self.receive_node(source_rank),
                TAG_TIME_OUT => self.receive_time_out(source_rank),
                TAG_TIME_OUT_ACK => self.receive_time_out_ack(source_rank),
                TAG_TERMINATION_DETECTION => self.receive_termination_detection(source_rank),
                TAG_TERMINATE => self.receive_terminate(source_rank),
                _ => self.search.receive_message(source_rank, tag),
            }
        }
    }

    fn pop_node_and_depth(&mut self) -> Option<(Rc<N>, usize)> {
        if self.is_layered_turn {
            if self.current_depth > self.layered_open.len() - 1 {
                self.current_depth = 0;
            }

            let initial_depth = self.current_depth;

            loop {
                let result = if self.memory_monitoring_interval.is_some() {
                    let (result, closed) = open_list::pop_from_open_and_count_closed(
                        &mut self.layered_open[self.current_depth],
                        &self.model,
                        self.search.get_primal_bound(),
                    );
                    self.search.close_registry_entries(closed);
                    result
                } else {
                    open_list::pop_from_open(
                        &mut self.layered_open[self.current_depth],
                        &self.model,
                        self.search.get_primal_bound(),
                    )
                };
                self.current_depth += 1;

                if let Some(node) = result {
                    self.is_layered_turn = false;

                    return Some((node, self.current_depth - 1));
                } else {
                    if self.current_depth > self.layered_open.len() - 1 {
                        self.current_depth = 0;
                    }

                    if self.current_depth == initial_depth {
                        break;
                    }
                }
            }
        }

        self.is_layered_turn = true;

        if self.memory_monitoring_interval.is_some() {
            let (result, closed) = open_list::pop_from_open_with_depth_and_count_closed(
                &mut self.open,
                &self.model,
                self.search.get_primal_bound(),
            );
            self.search.close_registry_entries(closed);
            result
        } else {
            open_list::pop_from_open_with_depth(
                &mut self.open,
                &self.model,
                self.search.get_primal_bound(),
            )
        }
    }

    fn compute_local_dual_bound(&self) -> Option<T> {
        if N::ordered_by_bound() {
            self.open
                .peek()
                .and_then(|(node, _)| node.bound(&self.model))
        } else {
            None
        }
    }

    pub fn search(&mut self) -> (Solution<T, TransitionWithId<V>>, Vec<Statistics>) {
        #[cfg(feature = "operation-timing")]
        let _search_timer = Timer::start(Operation::SearchTotal, 1);
        let mut keep_buffer = vec![];
        let mut send_buffer = vec![];

        loop {
            {
                #[cfg(feature = "operation-timing")]
                let _timer = Timer::start(Operation::PhaseMonitoring, 1);
                self.record_memory_statistics_if_due();
            }
            {
                #[cfg(feature = "operation-timing")]
                let _timer = Timer::start(Operation::PhaseMessages, 1);
                self.process_message();
            }

            {
                #[cfg(feature = "operation-timing")]
                let _timer = Timer::start(Operation::PhaseControl, 1);
                if self.is_terminated {
                    break;
                }

                if self.communicator.rank() == self.search.get_root_rank() {
                    if !self.is_time_out && self.search.check_time_limit() {
                        self.is_time_out = true;
                        self.local_dual_bound = self.compute_local_dual_bound();
                        self.broadcast_time_out();
                    }

                    if self.is_time_out
                        && self.n_remaining_time_out_ack == 0
                        && !self.is_checking_termination
                    {
                        self.is_checking_termination = true;
                        let destination_rank =
                            (self.communicator.rank() + 1) % self.communicator.size();
                        self.node_communicator
                            .initiate_termination(destination_rank);
                    }
                }

                if self.is_time_out {
                    continue;
                }
            }

            let next = {
                #[cfg(feature = "operation-timing")]
                let _timer = Timer::start(Operation::PhaseSelection, self.open.len());
                self.pop_node_and_depth()
            };
            if let Some((node, depth)) = next {
                {
                    #[cfg(feature = "operation-timing")]
                    let _timer = Timer::start(Operation::PhaseExpansion, 1);
                    if let Some(statistics) = self.expansion_statistics.as_mut() {
                        statistics.record(depth, node.bound(&self.model), node.cost(&self.model));
                    }
                    self.search.expand(
                        node,
                        &mut self.registry,
                        &mut keep_buffer,
                        &mut send_buffer,
                    );
                }

                {
                    #[cfg(feature = "operation-timing")]
                    let _timer = Timer::start(Operation::PhaseSending, send_buffer.len());
                    for (destination_rank, successor) in send_buffer.drain(..) {
                        self.node_communicator
                            .send(destination_rank, &successor, depth + 1);
                    }
                }

                {
                    #[cfg(feature = "operation-timing")]
                    let _timer = Timer::start(Operation::PhaseEnqueue, keep_buffer.len());
                    for successor in keep_buffer.drain(..) {
                        open_list::push_primary(&mut self.open, (successor.clone(), depth + 1));

                        while depth + 1 >= self.layered_open.len() {
                            self.layered_open.push(BinaryHeap::new());
                        }

                        open_list::push_layered(&mut self.layered_open[depth + 1], successor);
                    }
                }
            } else {
                #[cfg(feature = "operation-timing")]
                let _timer = Timer::start(Operation::PhaseNoWork, 1);
                if self.communicator.rank() == self.search.get_root_rank()
                    && !self.is_checking_termination
                {
                    self.is_checking_termination = true;
                    let destination_rank =
                        (self.communicator.rank() + 1) % self.communicator.size();
                    self.node_communicator
                        .initiate_termination(destination_rank);
                }
            }
        }

        {
            #[cfg(feature = "operation-timing")]
            let _timer = Timer::start(Operation::PhaseBarrier, 1);
            self.communicator.barrier();
        }

        #[cfg(feature = "operation-timing")]
        let _finalize_timer = Timer::start(Operation::PhaseFinalize, 1);
        let (mut solution, statistics) = self.search.finalize(self.local_dual_bound);

        if self.is_time_out {
            solution.time_out = true;
            solution.is_optimal = false;
            solution.is_infeasible = false;
        } else {
            solution.is_optimal = solution.cost.is_some();
            solution.is_infeasible = solution.cost.is_none();

            if solution.is_optimal {
                solution.best_bound = solution.cost;
            }
        }

        if !solution.is_optimal && solution.cost.is_some() && solution.cost == solution.best_bound {
            solution.is_optimal = true;
        }

        solution.time = self.search.elapsed_time();

        (solution, statistics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "memory-statistics")]
    use crate::DistributedFNode;
    #[cfg(feature = "memory-statistics")]
    use dypdl_heuristic_search::search_algorithm::data_structure::StateInformation;

    #[cfg(feature = "memory-statistics")]
    fn memory_test_state(signature: Integer, resource: Integer) -> State {
        State {
            signature_variables: SignatureVariables {
                integer_variables: vec![signature],
                ..Default::default()
            },
            resource_variables: ResourceVariables {
                integer_variables: vec![resource],
                ..Default::default()
            },
        }
    }

    #[cfg(feature = "memory-statistics")]
    fn memory_test_model() -> Rc<Model> {
        let mut model = Model::default();
        model.state_metadata.integer_variable_names = vec!["signature".into()];
        model.state_metadata.integer_resource_variable_names = vec!["resource".into()];
        model.state_metadata.integer_less_is_better = vec![true];
        model.target = memory_test_state(0, 0);
        Rc::new(model)
    }

    #[cfg(feature = "memory-statistics")]
    fn memory_test_node(
        signature: Integer,
        resource: Integer,
        cost: Integer,
    ) -> DistributedFNode<Integer> {
        DistributedFNode::new(
            StateInRegistry::from(memory_test_state(signature, resource)),
            cost,
            0,
            -cost,
            Rc::new(DistributedTransitionIdChain::default()),
        )
    }

    #[cfg(feature = "memory-statistics")]
    #[test]
    fn shared_signatures_and_dominated_nodes_in_two_queues() {
        let model = memory_test_model();
        let layout = HdHacSearchNodeMemoryLayout::with_model::<DistributedFNode<Integer>>(&model);
        let mut registry = StateRegistry::new(model);
        let counter = Rc::new(LiveNodeCounter::default());
        let mut primary = Vec::new();
        let mut layered = Vec::new();
        for (resource, cost) in [(1, 10), (2, 5)] {
            let node = registry
                .insert(memory_test_node(7, resource, cost))
                .information
                .unwrap();
            node.track_memory(&counter);
            primary.push(node.clone());
            layered.push(node);
        }
        assert_eq!(registry.signature_count(), 1);
        assert_eq!(counter.statistics().live, 2);
        assert!(Rc::ptr_eq(
            &primary[0].state().signature_variables,
            &primary[1].state().signature_variables
        ));
        assert!(registry
            .insert(memory_test_node(7, 3, 20))
            .information
            .is_none());
        assert_eq!(registry.signature_count(), 1);

        let result = registry.insert(memory_test_node(7, 0, 1));
        let replacement = result.information.unwrap();
        replacement.track_memory(&counter);
        assert_eq!(result.dominated.len(), 2);
        for old in &result.dominated {
            old.close();
        }
        drop(result.dominated);
        assert_eq!(counter.statistics().live, 3);
        assert_eq!(registry.signature_count(), 1);
        let (private, shared, storage) =
            layout.retained_components(3, registry.signature_count(), 1);
        assert_eq!(private, 3 * layout.private_node_and_resource_bytes);
        assert_eq!(shared, layout.shared_signature_bytes);
        assert_eq!(
            storage,
            layout.registry_signature_entry_bytes + layout.registry_node_reference_bytes
        );
        primary.clear();
        assert_eq!(counter.statistics().live, 3);
        layered.clear();
        assert_eq!(counter.statistics().live, 1);
        assert_eq!(counter.statistics().dropped, 2);

        let separate = registry
            .insert(memory_test_node(8, 0, 1))
            .information
            .unwrap();
        separate.track_memory(&counter);
        assert_eq!(registry.signature_count(), 2);
        drop(separate);
        drop(replacement);
        registry.clear();
        assert_eq!(counter.statistics().live, 0);
        assert_eq!(registry.signature_count(), 0);
    }

    #[cfg(feature = "memory-statistics")]
    #[test]
    fn bound_pruned_replacement_retains_the_signature_key() {
        let mut registry = StateRegistry::new(memory_test_model());
        let counter = Rc::new(LiveNodeCounter::default());
        let queue_node = registry
            .insert(memory_test_node(7, 1, 10))
            .information
            .unwrap();
        queue_node.track_memory(&counter);
        let result = registry.insert_with(
            StateInRegistry::from(memory_test_state(7, 0)),
            1,
            |_, _, _| None,
        );
        assert!(result.information.is_none());
        assert_eq!(result.dominated.len(), 1);
        drop(result);
        assert_eq!(registry.signature_count(), 1);
        assert_eq!(counter.statistics().live, 1);
        drop(queue_node);
        assert_eq!(counter.statistics().live, 0);
        assert_eq!(registry.signature_count(), 1);
        assert!(registry
            .insert_with(
                StateInRegistry::from(memory_test_state(8, 0)),
                1,
                |_, _, _| None
            )
            .information
            .is_none());
        assert_eq!(registry.signature_count(), 1);
    }

    #[test]
    fn parse_process_memory_from_linux_status() {
        let status = "Name:\ttest\nVmSize:\t1000 kB\nVmHWM:\t750 kB\nVmRSS:\t500 kB\n";

        assert_eq!(
            parse_linux_process_memory(status),
            ProcessMemory {
                resident_bytes: Some(500 * 1024),
                peak_resident_bytes: Some(750 * 1024),
                virtual_bytes: Some(1000 * 1024),
            }
        );
    }

    #[test]
    fn compute_state_variable_payload_bytes() {
        let mut set = Set::with_capacity(70);
        set.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set],
                vector_variables: vec![vec![1, 2, 3]],
                element_variables: vec![1],
                integer_variables: vec![2],
                continuous_variables: vec![3.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4],
                integer_variables: vec![5],
                continuous_variables: vec![6.0],
            },
        };
        let expected = mem::size_of::<Set>()
            + mem::size_of_val(state.signature_variables.set_variables[0].as_slice())
            + mem::size_of::<Vector>()
            + 3 * mem::size_of::<Element>()
            + 2 * mem::size_of::<Element>()
            + 2 * mem::size_of::<Integer>()
            + 2 * mem::size_of::<Continuous>();

        assert_eq!(state_variable_payload_bytes(&state), expected);
    }
}
