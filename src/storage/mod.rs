pub mod utils; //IMP -> uncomment this later just using rn for testing in main.rs
// mod utils;
use utils::{DataPtr, DataType, Device};

/*
Important stuff:

- DataPtr: size allocated on memory to this pointer ~24 bytes

*/


pub struct Storage {
    pub data_ptr_ : DataPtr, // data_ptr_ is a slice of vector according to rust.
    pub size_ : usize, // using usize instead of u32/u64 to be platform agnostic.
    data_type_ : DataType, // Datatype is a enum.
    device_ : Device // Device is a enum.
}

impl Storage {
    pub fn new(size: usize, data_type: DataType, device: Device) -> Self{
        let element_size: usize = match data_type {
            DataType::Float32 => std::mem::size_of::<f32>(),
            DataType::Float64 => std::mem::size_of::<f64>(),
            DataType::Int32 => std::mem::size_of::<i32>(),
            DataType::Int64 => std::mem::size_of::<i64>(),
        };// returns the size in bytes
        println!("The element_size assigned: {}", element_size);
        let buffer_size: usize = size * element_size; // returns the total buffer size in bytes
        println!("The buffer_size assigned: {}", buffer_size);
        Self {
            data_ptr_: DataPtr::new(Vec::with_capacity(buffer_size)), //TODO: Zero-initialized buffer using a idk what smart pointer??
            size_ : size, // Number of elements, not bytes
            data_type_: data_type,
            device_: device,
        }
    }
}