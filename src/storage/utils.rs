use std::sync::Arc;

// A smart pointer ?? shared data pointer
pub type DataPtr = Arc<Vec<usize>>; // Alias for Arc<Vec<usize>>

// enum for Device used in defining storage struct 
#[derive(Debug)]
pub enum Device {
    CPU, 
    CUDA,
    // Add more devices if needed
}

// enum for Datatype used in defining storage struct 
#[derive(Debug)]
pub enum DataType {
    Int32,
    Int64,
    Float32,
    Float64,
    // Add more types if needed
}
