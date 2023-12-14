/// A trait for converting a value into a tensor.
pub trait IntoTensor<T> {
    /// Converts the value into a tensor.
    fn into_tensor(self) -> Tensor<T>;
}

/// A tensor data structure that holds data and shape information.
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    /// The data contained in the tensor.
    pub(crate) data: T,
    /// The shape of the tensor.
    pub(crate) shape: Vec<usize>,
}

impl<T> Tensor<T> {
    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the data contained in the tensor.
    pub fn data(&self) -> &T {
        &self.data
    }
}