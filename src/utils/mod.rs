// enum for Device used in defining storage struct 
#[derive(Debug)]
pub enum Device {
    CPU, 
    CUDA,
    // Add more devices if needed
}

// enum for Datatype used in defining storage struct 
#[derive(Debug, Clone, PartialEq)] 
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
}

impl DataType {
    /// Returns the size of the data type in bytes.
    pub fn size_of(&self) -> usize {
        match self {
            DataType::Float32 => std::mem::size_of::<f32>(),
            DataType::Float64 => std::mem::size_of::<f64>(),
            DataType::Int32 => std::mem::size_of::<i32>(),
            DataType::Int64 => std::mem::size_of::<i64>(),
        }
    }// returns the size in bytes
}
