use std::collections::{BinaryHeap, VecDeque};
use std::fmt::Display;
use std::rc::Rc;

use dypdl::prelude::*;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::search_algorithm::data_structure;

use crate::bfs_node_with_distributed_id_chain::BfsNodeWithDistributedIdChain;

pub fn pop_from_queue<T, N>(
    open: &mut VecDeque<Rc<N>>,
    model: &Model,
    primal_bound: Option<T>,
) -> Option<Rc<N>>
where
    T: Numeric + Display,
    N: BfsNodeWithDistributedIdChain<T>,
{
    while let Some(node) = open.pop_front() {
        if node.is_closed() {
            continue;
        }

        node.close();

        if node.bound(model).map_or(false, |bound| {
            data_structure::exceed_bound(model, bound, primal_bound)
        }) {
            continue;
        }

        return Some(node);
    }

    None
}

pub fn pop_from_open<T, N>(
    open: &mut BinaryHeap<Rc<N>>,
    model: &Model,
    primal_bound: Option<T>,
) -> Option<Rc<N>>
where
    T: Numeric + Display,
    N: BfsNodeWithDistributedIdChain<T>,
{
    pop_from_open_and_count_closed(open, model, primal_bound).0
}

pub fn pop_from_open_and_count_closed<T, N>(
    open: &mut BinaryHeap<Rc<N>>,
    model: &Model,
    primal_bound: Option<T>,
) -> (Option<Rc<N>>, usize)
where
    T: Numeric + Display,
    N: BfsNodeWithDistributedIdChain<T>,
{
    let mut closed = 0;

    while let Some(node) = open.pop() {
        if node.is_closed() {
            continue;
        }

        node.close();
        closed += 1;

        if node.bound(model).map_or(false, |bound| {
            data_structure::exceed_bound(model, bound, primal_bound)
        }) {
            if N::ordered_by_bound() {
                open.clear();

                return (None, closed);
            }

            continue;
        }

        return (Some(node), closed);
    }

    (None, closed)
}

pub fn pop_from_open_with_depth<T, N>(
    open: &mut BinaryHeap<(Rc<N>, usize)>,
    model: &Model,
    primal_bound: Option<T>,
) -> Option<(Rc<N>, usize)>
where
    T: Numeric + Display,
    N: BfsNodeWithDistributedIdChain<T>,
{
    pop_from_open_with_depth_and_count_closed(open, model, primal_bound).0
}

pub fn pop_from_open_with_depth_and_count_closed<T, N>(
    open: &mut BinaryHeap<(Rc<N>, usize)>,
    model: &Model,
    primal_bound: Option<T>,
) -> (Option<(Rc<N>, usize)>, usize)
where
    T: Numeric + Display,
    N: BfsNodeWithDistributedIdChain<T>,
{
    let mut closed = 0;

    while let Some((node, depth)) = open.pop() {
        if node.is_closed() {
            continue;
        }

        node.close();
        closed += 1;

        if node.bound(model).map_or(false, |bound| {
            data_structure::exceed_bound(model, bound, primal_bound)
        }) {
            if N::ordered_by_bound() {
                open.clear();

                return (None, closed);
            }

            continue;
        }

        return (Some((node, depth)), closed);
    }

    (None, closed)
}

#[cfg(test)]
mod tests {
    use data_structure::StateInRegistry;
    use dypdl_heuristic_search::search_algorithm::data_structure::StateInformation;
    use std::cell::Cell;

    use crate::distributed_id_chain::{
        DistributedTransitionIdChain, GeRcDistributedTransitionIdChain,
        GetDistributedTransitionIdChain,
    };

    use super::*;

    #[derive(Debug)]
    struct MockGNode {
        state: StateInRegistry,
        g: Integer,
        closed: Cell<bool>,
        transition_id_chain: Rc<DistributedTransitionIdChain>,
    }

    impl MockGNode {
        fn new(g: Integer) -> Self {
            Self {
                state: StateInRegistry::default(),
                g,
                closed: Cell::new(false),
                transition_id_chain: Rc::new(DistributedTransitionIdChain::default()),
            }
        }
    }

    impl PartialEq for MockGNode {
        fn eq(&self, other: &Self) -> bool {
            self.g == other.g
        }
    }

    impl Eq for MockGNode {}

    impl Ord for MockGNode {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other.g.cmp(&self.g)
        }
    }

    impl PartialOrd for MockGNode {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl StateInformation<Integer> for MockGNode {
        fn state(&self) -> &StateInRegistry {
            &self.state
        }

        fn state_mut(&mut self) -> &mut StateInRegistry {
            &mut self.state
        }

        fn cost(&self, _model: &Model) -> Integer {
            self.g
        }

        fn bound(&self, _model: &Model) -> Option<Integer> {
            None
        }

        fn is_closed(&self) -> bool {
            self.closed.get()
        }

        fn close(&self) {
            self.closed.set(true);
        }
    }

    impl GetDistributedTransitionIdChain for MockGNode {
        fn get_distributed_transition_id_chain(&self) -> &DistributedTransitionIdChain {
            &self.transition_id_chain
        }
    }

    impl GeRcDistributedTransitionIdChain for MockGNode {
        fn get_rc_distributed_transition_id_chain(&self) -> &Rc<DistributedTransitionIdChain> {
            &self.transition_id_chain
        }
    }

    impl BfsNodeWithDistributedIdChain<Integer> for MockGNode {
        fn ordered_by_bound() -> bool {
            false
        }
    }

    #[derive(Debug)]
    struct MockFNode {
        node: MockGNode,
        h: Integer,
    }

    impl MockFNode {
        fn new(g: Integer, h: Integer) -> Self {
            Self {
                node: MockGNode::new(g),
                h,
            }
        }
    }

    impl PartialEq for MockFNode {
        fn eq(&self, other: &Self) -> bool {
            self.h == other.h && self.node == other.node
        }
    }

    impl Eq for MockFNode {}

    impl Ord for MockFNode {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match (other.node.g + other.h).cmp(&(self.node.g + self.h)) {
                std::cmp::Ordering::Equal => other.h.cmp(&self.h),
                ordering => ordering,
            }
        }
    }

    impl PartialOrd for MockFNode {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl StateInformation<Integer> for MockFNode {
        fn state(&self) -> &StateInRegistry {
            self.node.state()
        }

        fn state_mut(&mut self) -> &mut StateInRegistry {
            self.node.state_mut()
        }

        fn cost(&self, model: &Model) -> Integer {
            self.node.cost(model)
        }

        fn bound(&self, _model: &Model) -> Option<Integer> {
            Some(self.node.g + self.h)
        }

        fn is_closed(&self) -> bool {
            self.node.is_closed()
        }

        fn close(&self) {
            self.node.close()
        }
    }

    impl GetDistributedTransitionIdChain for MockFNode {
        fn get_distributed_transition_id_chain(&self) -> &DistributedTransitionIdChain {
            self.node.get_distributed_transition_id_chain()
        }
    }

    impl GeRcDistributedTransitionIdChain for MockFNode {
        #[inline]
        fn get_rc_distributed_transition_id_chain(&self) -> &Rc<DistributedTransitionIdChain> {
            self.node.get_rc_distributed_transition_id_chain()
        }
    }

    impl BfsNodeWithDistributedIdChain<Integer> for MockFNode {
        fn ordered_by_bound() -> bool {
            true
        }
    }

    #[derive(Debug)]
    struct MockWeightedFNode {
        node: MockFNode,
    }

    impl MockWeightedFNode {
        fn new(g: Integer, h: Integer) -> Self {
            Self {
                node: MockFNode::new(g, h),
            }
        }
    }

    impl PartialEq for MockWeightedFNode {
        fn eq(&self, other: &Self) -> bool {
            self.node == other.node
        }
    }

    impl Eq for MockWeightedFNode {}

    impl Ord for MockWeightedFNode {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match (other.node.node.g + 2 * other.node.h).cmp(&(self.node.node.g + 2 * self.node.h))
            {
                std::cmp::Ordering::Equal => other.node.h.cmp(&self.node.h),
                ordering => ordering,
            }
        }
    }

    impl PartialOrd for MockWeightedFNode {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl StateInformation<Integer> for MockWeightedFNode {
        fn state(&self) -> &StateInRegistry {
            self.node.state()
        }

        fn state_mut(&mut self) -> &mut StateInRegistry {
            self.node.state_mut()
        }

        fn cost(&self, model: &Model) -> Integer {
            self.node.cost(model)
        }

        fn bound(&self, model: &Model) -> Option<Integer> {
            self.node.bound(model)
        }

        fn is_closed(&self) -> bool {
            self.node.is_closed()
        }

        fn close(&self) {
            self.node.close()
        }
    }

    impl GetDistributedTransitionIdChain for MockWeightedFNode {
        fn get_distributed_transition_id_chain(&self) -> &DistributedTransitionIdChain {
            self.node.get_distributed_transition_id_chain()
        }
    }

    impl GeRcDistributedTransitionIdChain for MockWeightedFNode {
        #[inline]
        fn get_rc_distributed_transition_id_chain(&self) -> &Rc<DistributedTransitionIdChain> {
            self.node.get_rc_distributed_transition_id_chain()
        }
    }

    impl BfsNodeWithDistributedIdChain<Integer> for MockWeightedFNode {
        fn ordered_by_bound() -> bool {
            false
        }
    }

    #[test]
    fn pop_from_open_ordered_by_g() {
        let model = Model::default();
        let mut open = BinaryHeap::new();

        let node1 = Rc::new(MockGNode::new(5));
        open.push(node1);
        let node2 = Rc::new(MockGNode::new(2));
        node2.close();
        open.push(node2);
        let node3 = Rc::new(MockGNode::new(7));
        open.push(node3);
        let node4 = Rc::new(MockGNode::new(1));
        open.push(node4);
        let node5 = Rc::new(MockGNode::new(3));
        node5.close();
        open.push(node5);
        let node6 = Rc::new(MockGNode::new(4));
        open.push(node6);
        let node7 = Rc::new(MockGNode::new(6));
        node7.close();
        open.push(node7);

        let node = pop_from_open(&mut open, &model, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.g, 1);
        assert!(node.is_closed());

        let node = pop_from_open(&mut open, &model, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.g, 4);
        assert!(node.is_closed());

        let node = pop_from_open(&mut open, &model, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.g, 5);
        assert!(node.is_closed());

        let node = pop_from_open(&mut open, &model, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.g, 7);
        assert!(node.is_closed());

        let node = pop_from_open(&mut open, &model, None);
        assert!(node.is_none());
        assert!(open.is_empty());
    }

    #[test]
    fn pop_from_open_ordered_by_f() {
        let model = Model::default();
        let mut open = BinaryHeap::new();

        let node1 = Rc::new(MockFNode::new(2, 3));
        open.push(node1);
        let node2 = Rc::new(MockFNode::new(1, 1));
        node2.close();
        open.push(node2);
        let node3 = Rc::new(MockFNode::new(2, 5));
        open.push(node3);
        let node4 = Rc::new(MockFNode::new(0, 1));
        open.push(node4);
        let node5 = Rc::new(MockFNode::new(1, 2));
        node5.close();
        open.push(node5);
        let node6 = Rc::new(MockFNode::new(1, 3));
        open.push(node6);
        let node7 = Rc::new(MockFNode::new(3, 3));
        node7.close();
        open.push(node7);

        let node = pop_from_open(&mut open, &model, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.node.g, 0);
        assert_eq!(node.h, 1);
        assert!(node.is_closed());

        let node = pop_from_open(&mut open, &model, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.node.g, 1);
        assert_eq!(node.h, 3);
        assert!(node.is_closed());

        let node = pop_from_open(&mut open, &model, Some(5));
        assert!(node.is_none());
        assert!(open.is_empty());
    }

    #[test]
    fn pop_from_open_ordered_by_weighted_f() {
        let model = Model::default();
        let mut open = BinaryHeap::new();

        let node1 = Rc::new(MockWeightedFNode::new(2, 3));
        open.push(node1);
        let node2 = Rc::new(MockWeightedFNode::new(1, 1));
        node2.close();
        open.push(node2);
        let node3 = Rc::new(MockWeightedFNode::new(2, 5));
        open.push(node3);
        let node4 = Rc::new(MockWeightedFNode::new(0, 1));
        open.push(node4);
        let node5 = Rc::new(MockWeightedFNode::new(1, 2));
        node5.close();
        open.push(node5);
        let node6 = Rc::new(MockWeightedFNode::new(1, 3));
        open.push(node6);
        let node7 = Rc::new(MockWeightedFNode::new(3, 3));
        node7.close();
        open.push(node7);

        let node = pop_from_open(&mut open, &model, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.node.node.g, 0);
        assert_eq!(node.node.h, 1);
        assert!(node.is_closed());

        let node = pop_from_open(&mut open, &model, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.node.node.g, 1);
        assert_eq!(node.node.h, 3);
        assert!(node.is_closed());

        let node = pop_from_open(&mut open, &model, Some(6));
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.node.node.g, 2);
        assert_eq!(node.node.h, 3);
        assert!(node.is_closed());

        let node = pop_from_open(&mut open, &model, Some(6));
        assert!(node.is_none());
        assert!(open.is_empty());
    }

    #[test]
    fn pop_from_open_with_depth_ordered_by_g() {
        let model = Model::default();
        let mut open = BinaryHeap::new();

        let node1 = Rc::new(MockGNode::new(5));
        open.push((node1, 1));
        let node2 = Rc::new(MockGNode::new(2));
        node2.close();
        open.push((node2, 2));
        let node3 = Rc::new(MockGNode::new(7));
        open.push((node3, 3));
        let node4 = Rc::new(MockGNode::new(1));
        open.push((node4, 4));
        let node5 = Rc::new(MockGNode::new(3));
        node5.close();
        open.push((node5, 5));
        let node6 = Rc::new(MockGNode::new(4));
        open.push((node6, 6));
        let node7 = Rc::new(MockGNode::new(6));
        node7.close();
        open.push((node7, 7));

        let result = pop_from_open_with_depth(&mut open, &model, None);
        assert!(result.is_some());
        let (node, depth) = result.unwrap();
        assert_eq!(node.g, 1);
        assert!(node.is_closed());
        assert_eq!(depth, 4);

        let result = pop_from_open_with_depth(&mut open, &model, None);
        assert!(result.is_some());
        let (node, depth) = result.unwrap();
        assert_eq!(node.g, 4);
        assert!(node.is_closed());
        assert_eq!(depth, 6);

        let result = pop_from_open_with_depth(&mut open, &model, None);
        assert!(result.is_some());
        let (node, depth) = result.unwrap();
        assert_eq!(node.g, 5);
        assert!(node.is_closed());
        assert_eq!(depth, 1);

        let result = pop_from_open_with_depth(&mut open, &model, None);
        assert!(result.is_some());
        let (node, depth) = result.unwrap();
        assert_eq!(node.g, 7);
        assert!(node.is_closed());
        assert_eq!(depth, 3);

        let result = pop_from_open_with_depth(&mut open, &model, None);
        assert!(result.is_none());
        assert!(open.is_empty());
    }

    #[test]
    fn pop_from_open_with_depth_ordered_by_f() {
        let model = Model::default();
        let mut open = BinaryHeap::new();

        let node1 = Rc::new(MockFNode::new(2, 3));
        open.push((node1, 1));
        let node2 = Rc::new(MockFNode::new(1, 1));
        node2.close();
        open.push((node2, 2));
        let node3 = Rc::new(MockFNode::new(2, 5));
        open.push((node3, 3));
        let node4 = Rc::new(MockFNode::new(0, 1));
        open.push((node4, 4));
        let node5 = Rc::new(MockFNode::new(1, 2));
        node5.close();
        open.push((node5, 5));
        let node6 = Rc::new(MockFNode::new(1, 3));
        open.push((node6, 6));
        let node7 = Rc::new(MockFNode::new(3, 3));
        node7.close();
        open.push((node7, 7));

        let result = pop_from_open_with_depth(&mut open, &model, None);
        assert!(result.is_some());
        let (node, depth) = result.unwrap();
        assert_eq!(node.node.g, 0);
        assert_eq!(node.h, 1);
        assert!(node.is_closed());
        assert_eq!(depth, 4);

        let result = pop_from_open_with_depth(&mut open, &model, None);
        assert!(result.is_some());
        let (node, depth) = result.unwrap();
        assert_eq!(node.node.g, 1);
        assert_eq!(node.h, 3);
        assert!(node.is_closed());
        assert_eq!(depth, 6);

        let result = pop_from_open_with_depth(&mut open, &model, Some(5));
        assert!(result.is_none());
        assert!(open.is_empty());
    }

    #[test]
    fn pop_from_open_with_depth_ordered_by_weighted_f() {
        let model = Model::default();
        let mut open = BinaryHeap::new();

        let node1 = Rc::new(MockWeightedFNode::new(2, 3));
        open.push((node1, 1));
        let node2 = Rc::new(MockWeightedFNode::new(1, 1));
        node2.close();
        open.push((node2, 2));
        let node3 = Rc::new(MockWeightedFNode::new(2, 5));
        open.push((node3, 3));
        let node4 = Rc::new(MockWeightedFNode::new(0, 1));
        open.push((node4, 4));
        let node5 = Rc::new(MockWeightedFNode::new(1, 2));
        node5.close();
        open.push((node5, 5));
        let node6 = Rc::new(MockWeightedFNode::new(1, 3));
        open.push((node6, 6));
        let node7 = Rc::new(MockWeightedFNode::new(3, 3));
        node7.close();
        open.push((node7, 7));

        let result = pop_from_open_with_depth(&mut open, &model, None);
        assert!(result.is_some());
        let (node, depth) = result.unwrap();
        assert_eq!(node.node.node.g, 0);
        assert_eq!(node.node.h, 1);
        assert!(node.is_closed());
        assert_eq!(depth, 4);

        let result = pop_from_open_with_depth(&mut open, &model, None);
        assert!(result.is_some());
        let (node, depth) = result.unwrap();
        assert_eq!(node.node.node.g, 1);
        assert_eq!(node.node.h, 3);
        assert!(node.is_closed());
        assert_eq!(depth, 6);

        let result = pop_from_open_with_depth(&mut open, &model, Some(6));
        assert!(result.is_some());
        let (node, depth) = result.unwrap();
        assert_eq!(node.node.node.g, 2);
        assert_eq!(node.node.h, 3);
        assert!(node.is_closed());
        assert_eq!(depth, 1);

        let result = pop_from_open_with_depth(&mut open, &model, Some(6));
        assert!(result.is_none());
        assert!(open.is_empty());
    }
}
