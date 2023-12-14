use std::ops::Mul;
use crate::{Tensor, IntoTensor};

/// Converts a `Vec<T>` into a `Tensor<Vec<T>>`.
///
/// This trait implementation allows a `Vec<T>` to be converted into a `Tensor<Vec<T>>` by providing
/// the `as_tensor` method. The resulting `Tensor` will have a shape of `[length_of_vec]`.
///
/// # Examples
///
/// ```
/// use rusty_core::{Tensor, IntoTensor};
///
/// let vec = vec![1, 2, 3];
/// let tensor = vec.into_tensor();
/// assert_eq!(tensor.shape(), vec![3]);
/// ```
impl<T> IntoTensor<Vec<T>> for Vec<T> {
    /// Converts the `Vec<T>` value into a `Tensor<Vec<T>>`.
    ///
    /// # Returns
    ///
    /// A `Tensor` with shape [n] containing the `Vec<T>` data.
    fn into_tensor(self) -> Tensor<Vec<T>> {
        let shape = vec![self.len()];

        Tensor {
            data: self,
            shape
        }
    }
}

impl Mul<Tensor<f64>> for Tensor<Vec<f64>> {
    type Output = Tensor<Vec<f64>>;

    /// Multiplies two tensors element-wise.
    ///
    /// This implementation allows multiplying a `Tensor<Vec<f64>>` with a `Tensor<f64>`. The two
    /// tensors must have compatible shapes for element-wise multiplication. The resulting tensor
    /// will have the same shape as the left-hand side tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_core::{Tensor, IntoTensor};
    ///
    /// let tensor1 = vec![1.0, 2.0, 3.0].into_tensor();
    /// let tensor2 = 2.0.into_tensor();
    /// let result = tensor1 * tensor2;
    /// assert_eq!(result.data(), &vec![2.0, 4.0, 6.0]);
    /// ```
    fn mul(self, rhs: Tensor<f64>) -> Self::Output {
        let mut result = self.clone();
        result.data = result.data.iter().map(|x| x * rhs.data).collect();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul() {
        // Test with positive numbers
        let tensor1 = vec![2.0, 3.0, 4.0].into_tensor();
        let tensor2 = 2.0.into_tensor();
        let result = tensor1 * tensor2;
        assert_eq!(result.data(), &vec![4.0, 6.0, 8.0]);

        // Test with negative numbers
        let tensor1 = vec![-2.0, -3.0, -4.0].into_tensor();
        let tensor2 = -2.0.into_tensor();
        let result = tensor1 * tensor2;
        assert_eq!(result.data(), &vec![4.0, 6.0, 8.0]);

        // Test with zeros
        let tensor1 = vec![0.0, 0.0, 0.0].into_tensor();
        let tensor2 = 2.0.into_tensor();
        let result = tensor1 * tensor2;
        assert_eq!(result.data(), &vec![0.0, 0.0, 0.0]);

        // Test with mixed positive and negative numbers
        let tensor1 = vec![2.0, -3.0, 4.0].into_tensor();
        let tensor2 = -2.0.into_tensor();
        let result = tensor1 * tensor2;
        assert_eq!(result.data(), &vec![-4.0, 6.0, -8.0]);
    }
}