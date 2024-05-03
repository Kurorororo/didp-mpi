use dypdl::prelude::*;
use mpi::{
    datatype::DatatypeRef,
    traits::Equivalence,
    {Address, Count},
};
use std::mem;
use zerocopy::{AsBytes, FromBytes};

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
}

impl StateSerializer {
    fn compute_n_blocks(bits: usize) -> usize {
        let (mut blocks, rem) = (bits / 32, bits % 32);
        blocks += (rem > 0) as usize;
        blocks
    }

    pub fn with_model(model: &Model) -> Self {
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

        let total_size = n_total_set_variable_blocks * mem::size_of::<u32>()
            + n_element_variables * mem::size_of::<Element>()
            + n_integer_variables * mem::size_of::<Integer>()
            + n_continuous_variables * mem::size_of::<Continuous>()
            + n_element_resource_variables * mem::size_of::<Element>()
            + n_integer_resource_variables * mem::size_of::<Integer>()
            + n_continuous_resource_variables * mem::size_of::<Continuous>();

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
        }
    }

    pub fn with_state<S: StateInterface>(state: &S) -> Self {
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

        let total_size = n_total_set_variable_blocks * mem::size_of::<u32>()
            + n_element_variables * mem::size_of::<Element>()
            + n_integer_variables * mem::size_of::<Integer>()
            + n_continuous_variables * mem::size_of::<Continuous>()
            + n_element_resource_variables * mem::size_of::<Element>()
            + n_integer_resource_variables * mem::size_of::<Integer>()
            + n_continuous_resource_variables * mem::size_of::<Continuous>();

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
    pub fn serialize_to<S: StateInterface>(&self, state: &S, buffer: &mut [u8]) {
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
    }

    /// Deserialize a state and its g-, h-, and f-values from a buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not large enough.
    pub fn deserialize<S: From<State>>(&self, buffer: &[u8]) -> S {
        let mut offset = 0;

        let set_variables = (0..self.n_set_variables)
            .map(|i| {
                let bits = self.each_set_variable_bits[i];
                let size = Self::compute_n_blocks(bits) * mem::size_of::<u32>();
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
                let size = mem::size_of::<usize>();
                let v = usize::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let integer_variables = (0..self.n_integer_variables)
            .map(|_| {
                let size = mem::size_of::<Integer>();
                let v = Integer::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let continuous_variables = (0..self.n_continuous_variables)
            .map(|_| {
                let size = mem::size_of::<Continuous>();
                let v = Continuous::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let element_resource_variables = (0..self.n_element_resource_variables)
            .map(|_| {
                let size = mem::size_of::<Element>();
                let v = Element::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let integer_resource_variables = (0..self.n_integer_resource_variables)
            .map(|_| {
                let size = mem::size_of::<Integer>();
                let v = Integer::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        let continuous_resource_variables = (0..self.n_continuous_resource_variables)
            .map(|_| {
                let size = mem::size_of::<Continuous>();
                let v = Continuous::read_from(&buffer[offset..offset + size]).unwrap();
                offset += size;
                v
            })
            .collect::<Vec<_>>();

        S::from(State {
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
        })
    }

    pub fn get_datatype_blocklengths(&self) -> [Count; 7] {
        [
            self.n_total_set_variable_blocks as Count,
            self.n_element_variables as Count,
            self.n_integer_variables as Count,
            self.n_continuous_variables as Count,
            self.n_element_resource_variables as Count,
            self.n_integer_resource_variables as Count,
            self.n_continuous_resource_variables as Count,
        ]
    }

    pub fn get_datatype_displacements(&self) -> [Address; 7] {
        let mut displacements = [0; 7];
        let mut offset = 0;
        displacements[0] = offset as Address;
        offset += self.n_total_set_variable_blocks * mem::size_of::<u32>();
        displacements[1] = offset as Address;
        offset += self.n_element_variables * mem::size_of::<Element>();
        displacements[2] = offset as Address;
        offset += self.n_integer_variables * mem::size_of::<Integer>();
        displacements[3] = offset as Address;
        offset += self.n_continuous_variables * mem::size_of::<Continuous>();
        displacements[4] = offset as Address;
        offset += self.n_element_resource_variables * mem::size_of::<Element>();
        displacements[5] = offset as Address;
        offset += self.n_integer_resource_variables * mem::size_of::<Integer>();
        displacements[6] = offset as Address;

        displacements
    }

    pub fn get_datatype_types(&self) -> [DatatypeRef<'static>; 7] {
        [
            u32::equivalent_datatype(),
            Element::equivalent_datatype(),
            Integer::equivalent_datatype(),
            Continuous::equivalent_datatype(),
            Element::equivalent_datatype(),
            Integer::equivalent_datatype(),
            Continuous::equivalent_datatype(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_serializer() {
        let (model, state) = create_model_and_state();

        let serializer = StateSerializer::with_state(&state);
        let mut buffer = vec![0u8; serializer.get_total_size()];
        serializer.serialize_to(&state, &mut buffer);

        let serializer = StateSerializer::with_model(&model);
        let reconstructed_state = serializer.deserialize::<State>(&buffer);

        assert_eq!(reconstructed_state, state);
    }
}
