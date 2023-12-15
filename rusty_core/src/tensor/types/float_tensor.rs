use std::ops::{Mul, Neg};
use crate::{Tensor, IntoTensor};

/// Converts a `f64` into a `Tensor<f64>`.
///
/// This trait implementation allows a `f64` to be converted into a `Tensor<64>` by providing
/// the `as_tensor` method. The resulting `Tensor` will have a shape of `[1]`.
///
/// # Examples
///
/// ```
/// use rusty_core::{Tensor, IntoTensor};
///
/// let scalar = 2.0;
/// let tensor = scalar.into_tensor();
/// assert_eq!(tensor.shape(), vec![]);
/// ```
impl IntoTensor<f64> for f64 {
    /// Converts the `f64` value into a `Tensor<f64>`.
    ///
    /// # Returns
    ///
    /// A `Tensor` with shape [1] containing the `f64` value.
    fn into_tensor(self) -> Tensor<f64> {
        Tensor {
            data: self,
            shape: vec![]
        }
    }
}

/// Negates a tensor.
impl Neg for Tensor<f64> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Tensor {
            data: -self.data,
            shape: self.shape
        }
    }
}

/// Operator implementation for multiplying two tensors.
///
/// This implementation allows multiplying a `Tensor<f64>` with a `Tensor<Vec<f64>>`.
impl Mul<Tensor<Vec<f64>>> for Tensor<f64> {
    /// The resulting type after multiplying two tensors.
    ///
    /// # Returns
    ///
    /// A `Tensor<Vec<f64>>` representing the result of the multiplication.
    type Output = Tensor<Vec<f64>>;

    /// Multiplies two tensors.
    ///
    /// # Arguments
    ///
    /// * `self` - The first tensor to multiply.
    /// * `rhs` - The second tensor to multiply.
    ///
    /// # Returns
    ///
    /// A `Tensor<Vec<f64>>` representing the result of the multiplication.
    fn mul(self, rhs: Tensor<Vec<f64>>) -> Self::Output {
        rhs * self
    }
}