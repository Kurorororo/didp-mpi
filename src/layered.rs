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

    pub fn get_mut_or_create<F>(&mut self, depth: usize, default: F) -> &mut T
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

    // pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
    //     self.deque
    //         .iter()
    //         .enumerate()
    //         .map(|(i, value)| (self.minimum_depth + i, value))
    // }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> {
        self.deque
            .iter_mut()
            .enumerate()
            .map(|(i, value)| (self.minimum_depth + i, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layered_push_pop() {
        let mut layered = Layered::new(0);

        assert!(!layered.is_empty());
        assert_eq!(layered.get(0), Some(&0));
        assert_eq!(layered.get(1), None);
        assert_eq!(layered.get(2), None);
        assert_eq!(layered.get(3), None);
        assert_eq!(layered.get(4), None);
        assert_eq!(layered.minimum_depth(), 0);

        let back = layered.get_mut_or_create(3, || 1);
        *back = 3;

        let element = layered.get_mut(2);
        assert!(element.is_some());
        let element = element.unwrap();
        *element = 2;

        assert_eq!(layered.get(0), Some(&0));
        assert_eq!(layered.get(1), Some(&1));
        assert_eq!(layered.get(2), Some(&2));
        assert_eq!(layered.get(3), Some(&3));
        assert_eq!(layered.get(4), None);
        assert_eq!(layered.minimum_depth(), 0);
        assert_eq!(layered.maximum_depth(), 3);
        assert!(!layered.is_empty());

        layered.pop(1);

        assert_eq!(layered.get(2), Some(&2));
        assert_eq!(layered.get(3), Some(&3));
        assert_eq!(layered.get(4), None);
        assert_eq!(layered.minimum_depth(), 2);
        assert_eq!(layered.maximum_depth(), 3);
        assert!(!layered.is_empty());

        layered.get_mut_or_create(3, || 1);

        assert_eq!(layered.get(2), Some(&2));
        assert_eq!(layered.get(3), Some(&3));
        assert_eq!(layered.get(4), None);
        assert_eq!(layered.minimum_depth(), 2);
        assert_eq!(layered.maximum_depth(), 3);
        assert!(!layered.is_empty());

        let mut sum = 0;
        let callback = |i: &mut _, _| sum += *i;
        layered.pop_with(3, callback);

        assert_eq!(layered.get(4), None);
        assert_eq!(layered.minimum_depth(), 4);
        assert!(layered.is_empty());
        assert_eq!(sum, 5);
    }
}
