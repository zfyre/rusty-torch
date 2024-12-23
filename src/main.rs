mod storage;
use storage::Storage;

// test imports
use storage::utils::{DataPtr, DataType, Device};

fn main() {
    let stg: Storage = Storage::new(10, DataType::Float32, Device::CPU);
    
    println!("Number of elements: {}", stg.size_);
    println!("vector of values: {:?}", stg.data_ptr_);
    println!("the size of vector of buffer assigned: {:?}", stg.data_ptr_.capacity());
}   
