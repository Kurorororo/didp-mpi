use dypdl::prelude::*;
use dypdl::variable_type::{Continuous, Integer, Numeric, OrderedContinuous, Set};
use mpi::datatype::DatatypeRef;
use mpi::traits::Equivalence;
use mpi::{Address, Count};
use std::mem::size_of;
use zerocopy::{AsBytes, FromBytes};

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

/// Singleton struct for serialization and deserialization of a state and its g-, h-, and f-values.
#[derive(Debug)]
pub struct StateSerializer {
    n_total_set_variable_blocks: usize,
    each_set_variable_bits: Vec<usize>,
    n_set_variables: usize,
    n_element_variables: usize,
    n_integer_variables: usize,
    n_continuous_variables: usize,
    n_element_resource_variables: usize,
    n_integer_resource_variables: usize,
    n_continuous_resource_variables: usize,
    total_size: usize,
    is_cost_type_float: bool,
}

impl StateSerializer {
    fn compute_n_blocks(bits: usize) -> usize {
        let (mut blocks, rem) = (bits / 32, bits % 32);
        blocks += (rem > 0) as usize;
        blocks
    }

    pub fn with_model<T: IsFloat>(model: &Model) -> Self {
        let metadata = &model.state_metadata;

        let n_set_variables = metadata.number_of_set_variables();
        let each_set_variable_bits = (0..n_set_variables)
            .map(|i| {
                let object_id = metadata.set_variable_to_object[i];
                metadata.object_numbers[object_id]
            })
            .collect::<Vec<_>>();
        let n_total_set_variable_blocks = (0..n_set_variables)
            .map(|i| {
                let object_id = metadata.set_variable_to_object[i];
                let bits = metadata.object_numbers[object_id];
                Self::compute_n_blocks(bits)
            })
            .sum::<usize>();
        let n_element_variables = metadata.number_of_element_variables();
        let n_integer_variables = metadata.number_of_integer_variables();
        let n_continuous_variables = metadata.number_of_continuous_variables();
        let n_element_resource_variables = metadata.number_of_element_resource_variables();
        let n_integer_resource_variables = metadata.number_of_integer_resource_variables();
        let n_continuous_resource_variables = metadata.number_of_continuous_resource_variables();

        let is_float = T::is_float();
        let mut total_size = n_total_set_variable_blocks * size_of::<u32>()
            + n_element_variables * size_of::<Element>()
            + n_integer_variables * size_of::<Integer>()
            + n_continuous_variables * size_of::<Continuous>()
            + n_element_resource_variables * size_of::<Element>()
            + n_integer_resource_variables * size_of::<Integer>()
            + n_continuous_resource_variables * size_of::<Continuous>();

        if is_float {
            total_size += 3 * size_of::<Continuous>();
        } else {
            total_size += 3 * size_of::<Integer>();
        }

        Self {
            n_total_set_variable_blocks,
            each_set_variable_bits,
            n_set_variables,
            n_element_variables,
            n_integer_variables,
            n_continuous_variables,
            n_element_resource_variables,
            n_integer_resource_variables,
            n_continuous_resource_variables,
            total_size,
            is_cost_type_float: is_float,
        }
    }

    pub fn with_state<S: StateInterface, T: IsFloat>(state: &S) -> Self {
        let n_set_variables = state.get_number_of_set_variables();
        let each_set_variable_bits = (0..n_set_variables)
            .map(|i| {
                let v = state.get_set_variable(i);
                v.len()
            })
            .collect::<Vec<_>>();
        let n_total_set_variable_blocks = (0..n_set_variables)
            .map(|i| {
                let v = state.get_set_variable(i);
                let bits = v.len();
                Self::compute_n_blocks(bits)
            })
            .sum::<usize>();
        let n_element_variables = state.get_number_of_element_variables();
        let n_integer_variables = state.get_number_of_integer_variables();
        let n_continuous_variables = state.get_number_of_continuous_variables();
        let n_element_resource_variables = state.get_number_of_element_resource_variables();
        let n_integer_resource_variables = state.get_number_of_integer_resource_variables();
        let n_continuous_resource_variables = state.get_number_of_continuous_resource_variables();

        let is_float = T::is_float();
        let mut total_size = n_total_set_variable_blocks * size_of::<u32>()
            + n_element_variables * size_of::<Element>()
            + n_integer_variables * size_of::<Integer>()
            + n_continuous_variables * size_of::<Continuous>()
            + n_element_resource_variables * size_of::<Element>()
            + n_integer_resource_variables * size_of::<Integer>()
            + n_continuous_resource_variables * size_of::<Continuous>();

        if is_float {
            total_size += 3 * size_of::<Continuous>();
        } else {
            total_size += 3 * size_of::<Integer>();
        }

        Self {
            n_total_set_variable_blocks,
            each_set_variable_bits,
            n_set_variables,
            n_element_variables,
            n_integer_variables,
            n_continuous_variables,
            n_element_resource_variables,
            n_integer_resource_variables,
            n_continuous_resource_variables,
            total_size,
            is_cost_type_float: is_float,
        }
    }

    /// Get the total size in bytes.
    pub fn get_total_size(&self) -> usize {
        self.total_size
    }

    /// Serialize a state and its g-, h-, and f-values to a buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not large enough.
    pub fn serialize_to<S: StateInterface, T: Numeric>(
        &self,
        state: &S,
        g: T,
        h: T,
        f: T,
        buffer: &mut [u8],
    ) {
        let mut offset = 0;

        for i in 0..state.get_number_of_set_variables() {
            let v = state.get_set_variable(i);
            let bytes = v.as_slice().as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;
        }

        for i in 0..state.get_number_of_element_variables() {
            let v = state.get_element_variable(i);
            let bytes = v.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;
        }

        for i in 0..state.get_number_of_integer_variables() {
            let v = state.get_integer_variable(i);
            let bytes = v.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;
        }

        for i in 0..state.get_number_of_continuous_variables() {
            let v = state.get_continuous_variable(i);
            let bytes = v.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;
        }

        for i in 0..state.get_number_of_element_resource_variables() {
            let v = state.get_element_resource_variable(i);
            let bytes = v.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;
        }

        for i in 0..state.get_number_of_integer_resource_variables() {
            let v = state.get_integer_resource_variable(i);
            let bytes = v.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;
        }

        for i in 0..state.get_number_of_continuous_resource_variables() {
            let v = state.get_continuous_resource_variable(i);
            let bytes = v.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;
        }

        if self.is_cost_type_float {
            let g = g.to_continuous();
            let bytes = g.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;

            let h = h.to_continuous();
            let bytes = h.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;

            let f = f.to_continuous();
            let bytes = f.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
        } else {
            let g = g.to_integer();
            let bytes = g.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;

            let h = h.to_integer();
            let bytes = h.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
            offset += size;

            let f = f.to_integer();
            let bytes = f.as_bytes();
            let size = bytes.len();
            buffer[offset..offset + size].copy_from_slice(bytes);
        }
    }

    /// Deserialize a state and its g-, h-, and f-values from a buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not large enough.
    pub fn deserialize<S: From<State>, T: Numeric>(&self, buffer: &[u8]) -> (S, T, T, T) {
        let mut offset = 0;

        let set_variables = (0..self.n_set_variables)
            .map(|i| {
                let bits = self.each_set_variable_bits[i];
                let size = Self::compute_n_blocks(bits) * size_of::<u32>();
                let mut v = Set::with_capacity(bits);
                v.as_mut_slice()
                    .as_bytes_mut()
                    .copy_from_slice(&buffer[offset..offset + size]);
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let element_variables = (0..self.n_element_variables)
            .map(|_| {
                let size = size_of::<usize>();
                let v = usize::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let integer_variables = (0..self.n_integer_variables)
            .map(|_| {
                let size = size_of::<Integer>();
                let v = Integer::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let continuous_variables = (0..self.n_continuous_variables)
            .map(|_| {
                let size = size_of::<Continuous>();
                let v = Continuous::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let element_resource_variables = (0..self.n_element_resource_variables)
            .map(|_| {
                let size = size_of::<Element>();
                let v = Element::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let integer_resource_variables = (0..self.n_integer_resource_variables)
            .map(|_| {
                let size = size_of::<Integer>();
                let v = Integer::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let continuous_resource_variables = (0..self.n_continuous_resource_variables)
            .map(|_| {
                let size = size_of::<Continuous>();
                let v = Continuous::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let (g, h, f) = if self.is_cost_type_float {
            let size = size_of::<Continuous>();
            let g = T::from(Continuous::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = size_of::<Continuous>();
            let h = T::from(Continuous::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = size_of::<Continuous>();
            let f = T::from(Continuous::read_from(&buffer[offset..offset + size]).unwrap());

            (g, h, f)
        } else {
            let size = size_of::<Integer>();
            let g = T::from(Integer::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = size_of::<Integer>();
            let h = T::from(Integer::read_from(&buffer[offset..offset + size]).unwrap());
            offset += size;

            let size = size_of::<Integer>();
            let f = T::from(Integer::read_from(&buffer[offset..offset + size]).unwrap());

            (g, h, f)
        };

        let state = S::from(State {
            signature_variables: SignatureVariables {
                set_variables,
                vector_variables: Vec::default(),
                element_variables,
                integer_variables,
                continuous_variables,
            },
            resource_variables: ResourceVariables {
                element_variables: element_resource_variables,
                integer_variables: integer_resource_variables,
                continuous_variables: continuous_resource_variables,
            },
        });

        (state, g, h, f)
    }

    pub fn get_f<T: IsFloat>(&self, data: &[u8]) -> T {
        if T::is_float() {
            let offset = self.total_size - size_of::<Continuous>();
            let size = size_of::<Continuous>();
            T::from(Continuous::read_from(&data[offset..offset + size]).unwrap())
        } else {
            let offset = self.total_size - size_of::<Integer>();
            let size = size_of::<Integer>();
            T::from(Integer::read_from(&data[offset..offset + size]).unwrap())
        }
    }

    pub fn get_datatype_blocklengths(&self) -> [Count; 8] {
        [
            self.n_total_set_variable_blocks as Count,
            self.n_element_variables as Count,
            self.n_integer_variables as Count,
            self.n_continuous_variables as Count,
            self.n_element_resource_variables as Count,
            self.n_integer_resource_variables as Count,
            self.n_continuous_resource_variables as Count,
            3,
        ]
    }

    pub fn get_datatype_displacement(&self) -> [Address; 8] {
        let mut displacement = [0; 8];
        let mut offset = 0;
        displacement[0] = offset as Address;
        offset += self.n_total_set_variable_blocks * size_of::<u32>();
        displacement[1] = offset as Address;
        offset += self.n_element_variables * size_of::<Element>();
        displacement[2] = offset as Address;
        offset += self.n_integer_variables * size_of::<Integer>();
        displacement[3] = offset as Address;
        offset += self.n_continuous_variables * size_of::<Continuous>();
        displacement[4] = offset as Address;
        offset += self.n_element_resource_variables * size_of::<Element>();
        displacement[5] = offset as Address;
        offset += self.n_integer_resource_variables * size_of::<Integer>();
        displacement[6] = offset as Address;
        offset += self.n_continuous_resource_variables * size_of::<Continuous>();
        displacement[7] = offset as Address;

        displacement
    }

    pub fn get_datatype_types(&self) -> [DatatypeRef<'static>; 8] {
        let cost_type = if self.is_cost_type_float {
            Continuous::equivalent_datatype()
        } else {
            Integer::equivalent_datatype()
        };

        [
            u32::equivalent_datatype(),
            Element::equivalent_datatype(),
            Integer::equivalent_datatype(),
            Continuous::equivalent_datatype(),
            Element::equivalent_datatype(),
            Integer::equivalent_datatype(),
            Continuous::equivalent_datatype(),
            cost_type,
        ]
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

    fn create_model_and_state() -> (Model, State) {
        let mut model = Model::default();

        let object_type_1 = model.add_object_type("object1", 4);
        assert!(object_type_1.is_ok());
        let object_type_1 = object_type_1.unwrap();

        let object_type_2 = model.add_object_type("object2", 10);
        assert!(object_type_2.is_ok());
        let object_type_2 = object_type_2.unwrap();

        let object_type_3 = model.add_object_type("object3", 7);
        assert!(object_type_3.is_ok());
        let object_type_3 = object_type_3.unwrap();

        let set1 = Set::with_capacity(4);
        let v = model.add_set_variable("set1", object_type_1, set1);
        assert!(v.is_ok());

        let mut set2 = Set::with_capacity(10);
        set2.set_range(1..3, true);
        set2.set_range(7..10, true);
        let v = model.add_set_variable("set2", object_type_2, set2);
        assert!(v.is_ok());

        let mut set3 = Set::with_capacity(7);
        set3.set_range(0..7, true);
        let v = model.add_set_variable("set3", object_type_3, set3);
        assert!(v.is_ok());

        let v = model.add_element_variable("element1", object_type_2, 5);
        assert!(v.is_ok());

        let v = model.add_element_variable("element2", object_type_3, 1);
        assert!(v.is_ok());

        let v = model.add_integer_variable("integer1", 0);
        assert!(v.is_ok());

        let v = model.add_integer_variable("integer2", Integer::MAX);
        assert!(v.is_ok());

        let v = model.add_integer_variable("integer3", 10);
        assert!(v.is_ok());

        let v = model.add_integer_variable("integer4", Integer::MIN);
        assert!(v.is_ok());

        let v = model.add_continuous_variable("continuous1", Continuous::MIN);
        assert!(v.is_ok());

        let v = model.add_continuous_variable("continuous2", std::f64::consts::PI);
        assert!(v.is_ok());

        let v = model.add_continuous_variable("continuous4", Continuous::MAX);
        assert!(v.is_ok());

        let v = model.add_element_resource_variable("element_resource1", object_type_3, true, 9);
        assert!(v.is_ok());

        let v = model.add_element_resource_variable("element_resource2", object_type_1, false, 0);
        assert!(v.is_ok());

        let v = model.add_element_resource_variable("element_resource3", object_type_2, true, 4);
        assert!(v.is_ok());

        let v = model.add_integer_resource_variable("integer_resource2", false, -50);
        assert!(v.is_ok());

        let v = model.add_integer_resource_variable("integer_resource3", false, Integer::MIN);
        assert!(v.is_ok());

        let v = model.add_continuous_resource_variable("continuous_resource1", true, 0.0);
        assert!(v.is_ok());

        let state = model.target.clone();

        (model, state)
    }

    #[test]
    fn test_serializer_integer() {
        let (model, state) = create_model_and_state();

        let serializer = StateSerializer::with_state::<State, Integer>(&state);
        let mut buffer = vec![0u8; serializer.get_total_size()];
        let g = 2;
        let h = 3;
        let f = 5;
        serializer.serialize_to(&state, g, h, f, &mut buffer);

        assert_eq!(serializer.get_f::<Integer>(&buffer), f);

        let serializer = StateSerializer::with_model::<Integer>(&model);
        let (reconstructed_state, reconstructed_g, reconstructed_h, reconstructed_f) =
            serializer.deserialize::<State, Integer>(&buffer);

        assert_eq!(reconstructed_state, state);
        assert_eq!(reconstructed_g, g);
        assert_eq!(reconstructed_h, h);
        assert_eq!(reconstructed_f, f);
    }

    #[test]
    fn test_serializer_ordered_continuous() {
        let (model, state) = create_model_and_state();

        let serializer = StateSerializer::with_state::<State, OrderedContinuous>(&state);
        let mut buffer = vec![0u8; serializer.get_total_size()];
        let g = OrderedContinuous::from(1.5);
        let h = OrderedContinuous::from(2.4);
        let f = OrderedContinuous::from(3.9);
        serializer.serialize_to(&state, g, h, f, &mut buffer);

        assert_eq!(serializer.get_f::<OrderedContinuous>(&buffer), f);

        let serializer = StateSerializer::with_model::<OrderedContinuous>(&model);
        let (reconstructed_state, reconstructed_g, reconstructed_h, reconstructed_f) =
            serializer.deserialize::<State, OrderedContinuous>(&buffer);

        assert_eq!(reconstructed_state, state);
        assert_eq!(reconstructed_g, g);
        assert_eq!(reconstructed_h, h);
        assert_eq!(reconstructed_f, f);
    }
}
