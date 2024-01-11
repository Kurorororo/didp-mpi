use dypdl::variable_type::{Continuous, Integer, Numeric, OrderedContinuous};

pub trait IsFloat: Numeric {
    fn is_float() -> bool;
}

impl IsFloat for Integer {
    fn is_float() -> bool {
        false
    }
}

impl IsFloat for Continuous {
    fn is_float() -> bool {
        true
    }
}

impl IsFloat for OrderedContinuous {
    fn is_float() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_is_float() {
        assert!(!Integer::is_float());
    }

    #[test]
    fn test_continuous_is_float() {
        assert!(Continuous::is_float());
    }

    #[test]
    fn test_ordered_continuous_is_float() {
        assert!(OrderedContinuous::is_float());
    }
}
