//! Lifetime accounting for accepted node objects, including lazy queue entries.
use std::cell::{Cell, OnceCell};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveNodeStatistics {
    pub live: usize,
    pub created: usize,
    pub dropped: usize,
}

#[derive(Debug, Default)]
pub struct LiveNodeCounter {
    statistics: Cell<LiveNodeStatistics>,
}

impl LiveNodeCounter {
    pub fn statistics(&self) -> LiveNodeStatistics {
        self.statistics.get()
    }

    fn created(&self) {
        let mut s = self.statistics.get();
        s.live = s.live.checked_add(1).expect("live node counter overflow");
        s.created = s
            .created
            .checked_add(1)
            .expect("created node counter overflow");
        self.statistics.set(s);
    }

    fn dropped(&self) {
        let mut s = self.statistics.get();
        s.live = s.live.checked_sub(1).expect("live node counter underflow");
        s.dropped = s
            .dropped
            .checked_add(1)
            .expect("dropped node counter overflow");
        self.statistics.set(s);
    }
}

/// Optional per-node handle; included in the node's measured inline size.
#[derive(Debug, Default)]
pub struct NodeMemoryToken {
    counter: OnceCell<Rc<LiveNodeCounter>>,
}

impl NodeMemoryToken {
    pub fn track(&self, counter: &Rc<LiveNodeCounter>) {
        if let Some(existing) = self.counter.get() {
            assert!(
                Rc::ptr_eq(existing, counter),
                "node belongs to another memory counter"
            );
        } else {
            self.counter
                .set(counter.clone())
                .expect("memory counter already set");
            counter.created();
        }
    }
}

impl Clone for NodeMemoryToken {
    fn clone(&self) -> Self {
        let token = Self::default();
        if let Some(counter) = self.counter.get() {
            token.track(counter);
        }
        token
    }
}

impl Drop for NodeMemoryToken {
    fn drop(&mut self) {
        if let Some(counter) = self.counter.get() {
            counter.dropped();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_queue_references_keep_a_removed_node_alive() {
        let counter = Rc::new(LiveNodeCounter::default());
        let node = Rc::new(NodeMemoryToken::default());
        node.track(&counter);
        node.track(&counter);
        let first_queue = node.clone();
        let second_queue = node.clone();
        drop(node);
        assert_eq!(counter.statistics().live, 1);
        drop(first_queue);
        assert_eq!(counter.statistics().live, 1);
        drop(second_queue);
        assert_eq!(
            counter.statistics(),
            LiveNodeStatistics {
                live: 0,
                created: 1,
                dropped: 1
            }
        );
    }

    #[test]
    fn object_clones_count_but_rc_clones_do_not() {
        let counter = Rc::new(LiveNodeCounter::default());
        let node = Rc::new(NodeMemoryToken::default());
        node.track(&counter);
        let reference = node.clone();
        assert_eq!(counter.statistics().live, 1);
        let object = node.as_ref().clone();
        assert_eq!(counter.statistics().live, 2);
        drop(object);
        drop(reference);
        drop(node);
        assert_eq!(
            counter.statistics(),
            LiveNodeStatistics {
                live: 0,
                created: 2,
                dropped: 2
            }
        );
    }

    #[test]
    fn an_untracked_clone_can_be_tracked_independently() {
        let counter = Rc::new(LiveNodeCounter::default());
        let original = NodeMemoryToken::default();
        let clone = original.clone();
        clone.track(&counter);
        drop(original);
        assert_eq!(counter.statistics().live, 1);
        drop(clone);
        assert_eq!(counter.statistics().live, 0);
    }
}
