use std::collections::VecDeque;

pub struct Layered<T> {
    deque: VecDeque<T>,
    minimum_depth: usize,
}

impl<T> Layered<T> {
    pub fn new(value: T) -> Self {
        Self {
            deque: VecDeque::from([value]),
            minimum_depth: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }

    pub fn minimum_depth(&self) -> usize {
        self.minimum_depth
    }

    pub fn maximum_depth(&self) -> usize {
        debug_assert!(!self.is_empty());

        self.minimum_depth + self.deque.len() - 1
    }

    pub fn get(&self, depth: usize) -> Option<&T> {
        self.deque.get(depth - self.minimum_depth)
    }

    pub fn get_mut(&mut self, depth: usize) -> Option<&mut T> {
        self.deque.get_mut(depth - self.minimum_depth)
    }

    pub fn push<F>(&mut self, depth: usize, default: F) -> &mut T
    where
        F: Fn() -> T,
    {
        while self.minimum_depth + self.deque.len() <= depth {
            self.deque.push_back(default());
        }

        self.get_mut(depth).unwrap()
    }

    pub fn pop(&mut self, depth: usize) {
        while self.minimum_depth <= depth {
            self.deque.pop_front();
            self.minimum_depth += 1;
        }
    }

    pub fn pop_with<F>(&mut self, depth: usize, mut callback: F)
    where
        F: FnMut(&mut T, usize),
    {
        while self.minimum_depth <= depth {
            if let Some(mut front) = self.deque.pop_front() {
                callback(&mut front, self.minimum_depth);
            }

            self.minimum_depth += 1;
        }
    }

    pub fn filter_map_min_depth<F, R>(&mut self, mut callback: F) -> Option<(R, usize)>
    where
        F: FnMut(&mut T) -> Option<R>,
    {
        for (d, value) in self.deque.iter_mut().enumerate() {
            if let Some(result) = callback(value) {
                return Some((result, self.minimum_depth + d));
            }
        }

        None
    }
}
