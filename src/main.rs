mod backend;
use backend::{backend::TensorBackend, Cpu_backend::CpuTensorBackend};


fn main() {
    let a = CpuTensorBackend::<f32>::new(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("{:?}", a);

    let v = a.get(&[1,1]);
    println!("{:?}", v);

    a.print();
    
}


fn iterate_display<T: Iterator> (itr: T)
where
    T::Item: std::fmt::Display,
{
    for i in itr.enumerate() {
        print!("{} ", i.1);
    }
}


#[cfg(test)]
mod tests {
    use crate::iterate_display;


    #[test]
    fn test_generic_iterator() {
        let a = (1, 2, 3);
        let b = [4, 5, 6];

        // iterate_display(a.iter()); // as a matter of fact iterators are not implemented by tuples
        iterate_display(b.iter());

    }

}
