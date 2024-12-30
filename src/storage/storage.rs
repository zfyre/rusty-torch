use crate::utils::{DataType, Device};


pub struct Storage {
    pub data_ptr_ : DataPtr, // data_ptr_ is a slice of vector according to rust.
    pub size_ : usize, // using usize instead of u32/u64 to be platform agnostic.
    data_type_ : DataType, // Datatype is a enum.
    device_ : Device // Device is a enum.
}
