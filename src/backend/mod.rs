// Tensor::<CpuRawTensor<f32>>::new(&[3, 2], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

pub mod backend{
    use std::usize;
    use crate::backend::Cpu_backend::CpuTensorBackend;
    
    // TODO: try explicitely creating every tensor to be mutable and hence then directly modifying the value of the RawTensor instead of borrow and then return new tensor
    // something like inplace could be implemented !! but let's go with sinplicity right now
    pub trait DataType{
        type Repr; // associated dtype f32 are unstable otw i'll use f32
        fn add(&self, other: &Self) -> Self::Repr;
        fn sub(&self, other: &Self) -> Self::Repr;
        fn mul(&self, other: &Self) -> Self::Repr;
        fn div(&self, other: &Self) -> Self::Repr;
    }


    impl DataType for f32 {
        type Repr = f32;
        fn add(&self, other: &Self) -> Self::Repr {
            self + other
        }
        fn sub(&self, other: &Self) -> Self::Repr {
            self - other
        }
        fn div(&self, other: &Self) -> Self::Repr {
            self / other
        }
        fn mul(&self, other: &Self) -> Self::Repr {
            self * other
        }
    }


    pub trait TensorBackend{
        type dtype;
        
        // creation of a raw-tensor on specified backend
        fn new(shape: &[usize], data: &[Self::dtype]) -> Self;
        
        // raw TensorOps
        
        // unary ops
        fn exp(&self) -> Self;
        fn log(&self) -> Self;
        fn abs(&self, dim: Option<&[usize]>) -> Self;

        // binary ops
        fn add(&self, other: &Self) -> Self;
        fn sub(&self, other: &Self) -> Self;
        fn mul(&self, other: &Self) -> Self;
        fn div(&self, other: &Self) -> Self;
        fn pow(&self, other: &Self) -> Self;
        fn eq(&self, other: &Self) -> Self;
        
        // this is for lets say returning a tensor with 1 where the condition is true else false
        fn apply_cond(&self, f: impl Fn(Self::dtype) -> Self::dtype) -> Self; // yohoo static dispatch over a function

        // reduction ops
        fn sum(&self, dim: &[usize]) -> Self;
        fn max(&self, dim: &[usize]) -> Self; // we cannot use dim as an iterator beacuse somethings like tuples have not 'em implemented
        fn min(&self, dim: &[usize]) -> Self;

        // shape movement ops
        fn permute(&self, dim: &[usize]) -> Self;
        fn reshape(&self, dim: &[usize]) -> Self;
        fn expand(&self, shape: &[usize]) -> Self;
        fn unsqueeze(&self, dim: &[usize]) -> Self;

        // modification ops
        fn pad(&self, dim : &[usize]) -> Self;
        fn crop(&self, dim : &[usize]) -> Self;


        // utility func
        fn shape(&self) -> &[usize];
        fn ravel(&self) -> Vec<Self::dtype>;

        // device onloading -> then from Cpu I can offload to oher devices
        fn to_Cpu(&self) -> CpuTensorBackend<Self::dtype> where Self::dtype: DataType;

    }

}

// ==================================================== x === x === x ================================================

pub mod Cpu_backend {

    use std::{clone, iter, rc::Rc};
    // use crate::backend::backend::TensorBackend;
    use crate::backend::strider::ShapeStrider;
    use super::{backend::{DataType, TensorBackend}, shape::Shape, strider::TensorIter};

    #[derive(Debug)]
    struct Buffer<T: DataType> { // right now it's just a wrapper class but can be extended later!!
        data: Vec<T>,
    }

    #[derive(Debug)]
    pub struct CpuTensorBackend<T: DataType>{
        buffer: Rc<Buffer<T>>,
        strider: ShapeStrider,   
    }
    pub struct CpuTensorIter<'a, T: DataType> {
        tensor: &'a CpuTensorBackend<T>,
        tensor_iter: TensorIter<'a>,
    }
    impl<'a, T: DataType + Copy> Iterator for CpuTensorIter<'a, T> {
        type Item = T;
        fn next(&mut self) -> Option<Self::Item> {
            self.tensor_iter.next()
                            .map(|index| self.tensor.buffer.data[self.tensor.strider.buffer_idx(&index)])
        }
    }

    impl<'a, T: DataType + Copy> IntoIterator for &'a CpuTensorBackend<T> {
        type Item = T;
        type IntoIter = CpuTensorIter<'a, T>;
        fn into_iter(self) -> Self::IntoIter {
            CpuTensorIter {
                tensor: self,
                tensor_iter : TensorIter::new(&self.strider)
            }
        }
    }
    // Implementations of CpuTensorBackend class
    impl<T: DataType + Copy> CpuTensorBackend<T> {
        
        // some utility functions for creations
        fn _new(shape: &[usize], data: &[T]) -> Self {
            let buffer = Rc::new(Buffer{
                data: data.to_owned()
            });
            let strider = ShapeStrider::new_default(shape);

            Self {
                buffer,
                strider
            }
        }

        // User Tensor API functions
        pub fn get(&self, index: &[usize]) -> T {
            let buff_idx = self.strider.buffer_idx(index);
            self.buffer.data[buff_idx]
        }  
        
        pub fn print(&self) -> () 
        where
            T: std::fmt::Debug,
        {
            for i in self.into_iter() {
                print!("{:?} ", i);
            }
        }

        // implementing map function than applied a lambda for each of the elements
        // this uses the iterator which further uses the strider
        
    }

    impl<T: DataType + Copy> TensorBackend for CpuTensorBackend<T> {

        type dtype = T;

        fn new(shape: &[usize], data: &[Self::dtype]) -> Self {
            Self::_new(shape, data)
        }
        fn exp(&self) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn log(&self) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        
        fn abs(&self, dim: Option<&[usize]>) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        
        fn add(&self, other: &Self) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn sub(&self, other: &Self) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn mul(&self, other: &Self) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn div(&self, other: &Self) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        
        fn crop(&self, dim : &[usize]) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn expand(&self, shape: &[usize]) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn max(&self, dim: &[usize]) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn min(&self, dim: &[usize]) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn pow(&self, other: &Self) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn pad(&self, dim : &[usize]) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn ravel(&self) -> Vec<Self::dtype> {
            self.buffer.data.clone()
        }
        fn permute(&self, dim: &[usize]) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn apply_cond(&self, f: impl Fn(Self::dtype) -> Self::dtype) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        } 
        fn eq(&self, other: &Self) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn reshape(&self, dim: &[usize]) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }fn sum(&self, dim: &[usize]) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn unsqueeze(&self, dim: &[usize]) -> Self {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn to_Cpu(&self) -> CpuTensorBackend<Self::dtype> where Self::dtype: DataType {
            Self::_new(self.shape(), &self.buffer.data)
        }
        fn shape(&self) -> &[usize] {
            self.strider.shape()
        }
        
    }
            
}

// ==================================================== x === x === x ================================================
    
pub mod strider {

    // Imports
    use crate::backend::shape::Shape;

    #[derive(Debug)]
    pub struct ShapeStrider { /* This Struct is just a wrapper on methods which are required during padding,
         resizing , permuations etc. to change the stride and shape!! */
        shape: Vec<usize>,
        stride: Vec<usize>,
        offset: usize
    }
    pub struct TensorIter<'a> { // iterates over the tensor indexes
        strider: &'a ShapeStrider,
        tensor_idx: Vec<usize>,
        exhausted: bool
    }// Note: Rust does'nt allow some fields of a struct to be mutable and others immutable!!, hence either the whole struct is mutable or immutable

    impl<'a> TensorIter<'a> {
        pub fn new(strider: &'a ShapeStrider) -> Self {
            let tensor_idx = vec![0; strider.shape().ndims()]; // initialzed with (0,0,0) for a 3d tensor
            let exhausted= !strider.is_valid_index(&tensor_idx); // creating a checker for the bounds of index using strider
            Self {
                strider,
                tensor_idx,
                exhausted
            }
        }
    }
    impl<'a> Iterator for TensorIter<'a> {
        type Item = Vec<usize>;
        fn next(&mut self) -> Option<Self::Item> { // A do {...} while { ... } type thing
            if self.exhausted {
                return None;
            }

            let res = self.tensor_idx.clone();
            for i in (0..self.strider.shape.ndims()).rev() {
                self.tensor_idx[i] += 1;
                if self.tensor_idx[i] < self.strider.shape[i] {
                    break;
                }
                self.tensor_idx[i] = 0;
            }
            self.exhausted = self.tensor_idx.iter().all(|e| *e == 0);

            return Some(res);
        }
    }

    impl ShapeStrider {

        fn empty() -> Self{
            Self {
                shape: vec![],
                stride: vec![],
                offset: 0,
            }
        }
        pub fn new_default(shape: &[usize]) -> Self {

            if shape.is_empty() {
                return Self::empty(); // basically needed during initialization and then modifying later!!
            }

            let mut stride = vec![1_usize; shape.len()];
            for i in (0..shape.len() - 1).rev() { // n-2 -> 0
                stride[i] = stride[i+1] * shape[i+1];
            }

            Self {
                shape: shape.to_owned(),
                stride: stride,
                offset: 0,
            }
        }
        pub fn buffer_idx(&self, index: &[usize]) -> usize {
            self.offset + index.iter()
                                .zip(self.stride.iter())
                                .map(|(&i, &s)| i*s)
                                .sum::<usize>()
                                
        }


        pub fn is_valid_index(&self, index: &[usize]) -> bool {
            // for (i, s) in index.iter().zip(self.shape.iter()) {
            //     if  i >= s || *i < 0 {
            //         return false;
            //     }
            // }
            // return true;

            // one liner
            !self.shape.is_empty() && index.iter().zip(self.shape.iter()).all(|(i, s)| i < s)
        }
    }
    impl Shape for ShapeStrider {
        fn shape(&self) -> &[usize] {
            &self.shape
        }
    }
}

// ==================================================== x === x === x ================================================

pub mod shape {
    pub trait Shape {
        fn shape(&self) -> &[usize]; // Any struct can Implement this !! and then we can provide default implementations for Vec<usize> and [usize] so that struct.shape().default_fn() can work!!

        // default implementations 
        fn ndims(&self) -> usize {
            self.shape().len()
        }
        fn size(&self) -> usize {
            if self.ndims() == 0 {
                return 0;
            }
            self.shape().iter().product()
        }
    }

    impl Shape for &[usize] {
        fn shape(&self) -> &[usize] {
            self
        }
    }
    impl Shape for Vec<usize> {
        fn shape(&self) -> &[usize] {
            self
        }
    }
}