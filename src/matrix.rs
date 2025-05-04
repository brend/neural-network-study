use rand::{Rng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A simple 2-dimensional matrix with basic operations
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Creates a new matrix with the given number of rows and columns,
    /// initialized to zero.
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols];
        Self { rows, cols, data }
    }

    /// Creates a new matrix with the given number of rows and columns,
    /// initialized with random values between -1.0 and 1.0.
    pub fn random(rng: &mut StdRng, rows: usize, cols: usize) -> Self {
        let data = (0..(rows * cols))
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        Self { rows, cols, data }
    }

    /// Creates a new matrix from a 2D vector.
    /// The outer vector represents the rows, and the inner vectors represent the columns.
    /// Panics if the inner vectors have different lengths.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        if data.len() != rows * cols {
            panic!("data length does not match row and col count")
        }
        Self { rows, cols, data }
    }

    /// Creates a new matrix from a column vector.
    pub fn from_col_vec(data: Vec<f64>) -> Self {
        let rows = data.len();
        let cols = 1;
        Self::from_vec(rows, cols, data)
    }

    /// Transposes the matrix.
    pub fn transpose(&self) -> Self {
        let mut transposed_data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed_data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Self::from_vec(self.cols, self.rows, transposed_data)
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the column at the given index as a vector.
    /// Panics if the index is out of bounds.
    pub fn col(&self, col: usize) -> Vec<f64> {
        if col >= self.cols {
            panic!("Index out of bounds");
        }
        (0..self.rows)
            .map(|i| self.data[i * self.cols + col])
            .collect()
    }

    /// Returns a reference to the data in the matrix.
    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }

    /// Returns a mutable reference to the data in the matrix.
    pub fn data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.data
    }

    /// Returns the value at the given row and column.
    /// Panics if the indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        self.data[row * self.cols + col]
    }

    /// Returns a mutable reference to the value at the given row and column.
    /// Panics if the indices are out of bounds.
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        &mut self.data[row * self.cols + col]
    }

    /// Sets the value at the given row and column.
    /// Panics if the indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        self.data[row * self.cols + col] = value;
    }

    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let index = i * self.cols + j;
                self.data[index] = f(self.data[index]);
            }
        }
    }

    pub fn hadamard_product(&mut self, other: &Matrix) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrices must have the same dimensions for Hadamard product");
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, self.get(i, j) * other.get(i, j));
            }
        }
    }

    fn multiply_matrix_parallelized(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Matrices have incompatible dimensions for multiplication");
        }

        let other_t = Arc::new(other.transpose()); // wrap in Arc
        let self_data = &self.data;
        let other_data = &other_t.data;
        let self_cols = self.cols;
        let other_cols = other.cols;

        let result_data: Vec<f64> = (0..self.rows)
            .into_par_iter()
            .flat_map_iter(|i| {
                (0..other_t.rows).map(move |j| {
                    let mut sum = 0.0;
                    let row_start = i * self_cols;
                    let col_start = j * self_cols;
                    for k in 0..self_cols {
                        sum += self_data[row_start + k] * other_data[col_start + k];
                    }
                    sum
                })
            })
            .collect();

        Matrix::from_vec(self.rows, other_cols, result_data)
    }

    fn multiply_matrix_naive(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Matrices have incompatible dimensions for multiplication");
        }

        let other_t = other.transpose(); // for better cache locality
        let mut result = Matrix::new(self.rows, other.cols);

        let self_data = &self.data;
        let other_data = &other_t.data;
        let result_data = &mut result.data;

        let m = self.rows;
        let n = self.cols;
        let p = other.cols;

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                let a_row = i * n;
                let b_row = j * n; // because other_t has shape [p x n]
                for k in 0..n {
                    sum += self_data[a_row + k] * other_data[b_row + k];
                }
                result_data[i * p + j] = sum;
            }
        }

        result
    }

    pub fn multiply_matrix(&self, other: &Matrix) -> Matrix {
        if self.rows * other.cols >= 128 * 128 {
            self.multiply_matrix_parallelized(other)
        } else {
            self.multiply_matrix_naive(other)
        }
    }
}

impl Add<&Matrix> for Matrix {
    type Output = Matrix;

    /// Adds two matrices together, component-wise.
    /// Panics if the matrices have different dimensions.
    fn add(self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrices must have the same dimensions to be added");
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) + other.get(i, j));
            }
        }
        result
    }
}

impl AddAssign<&Matrix> for Matrix {
    /// Adds another matrix to this matrix, component-wise.
    /// Panics if the matrices have different dimensions.
    /// This is an in-place operation.
    fn add_assign(&mut self, other: &Matrix) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrices must have the same dimensions to be added");
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, self.get(i, j) + other.get(i, j));
            }
        }
    }
}

impl Sub<&Matrix> for Matrix {
    type Output = Matrix;

    /// Subtracts another matrix from this matrix, component-wise.
    /// Panics if the matrices have different dimensions.
    fn sub(self, rhs: &Matrix) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Matrices must have the same dimensions to be subtracted");
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) - rhs.get(i, j));
            }
        }
        result
    }
}

impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, other: &Matrix) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrices must have the same dimensions to be added");
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, self.get(i, j) - other.get(i, j));
            }
        }
    }
}

impl Mul<f64> for Matrix {
    type Output = Matrix;

    /// Multiplies the matrix by a scalar.
    fn mul(self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i * self.cols + j] = self.data[i * self.cols + j] * scalar;
            }
        }
        result
    }
}

impl MulAssign<f64> for Matrix {
    /// Multiplies the matrix by a scalar in-place.
    fn mul_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i * self.cols + j] *= scalar;
            }
        }
    }
}

use std::sync::Arc;

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix {
        self.multiply_matrix(other)
    }
}

#[cfg(test)]
mod matrix_tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn it_works() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data().len(), 2 * 3);
    }

    #[test]
    fn it_creates_random_matrix() {
        let mut rng = StdRng::from_os_rng();
        let m = Matrix::random(&mut rng, 2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data.len(), 2 * 3);
        for i in 0..2 {
            for j in 0..3 {
                assert!(m.get(i, j) >= -1.0 && m.get(i, j) <= 1.0);
            }
        }
    }

    #[test]
    fn it_creates_a_matrix_from_a_vector() {
        let v = vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0];
        let m = Matrix::from_vec(2, 3, v.clone());
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data, v);
    }

    #[test]
    fn it_transposes_matrix() {
        let m = Matrix::from_vec(
            3,
            2,
            vec![
                /* row 0 */ 1.0, 2.0, /* row 1 */ 5.0, 3.0, /* row 2 */ 4.0, 6.0,
            ],
        );
        let transposed = m.transpose();
        assert_eq!(transposed.rows, 2);
        assert_eq!(transposed.cols, 3);
        assert_eq!(transposed.get(0, 0), 1.0);
        assert_eq!(transposed.get(0, 1), 5.0);
        assert_eq!(transposed.get(0, 2), 4.0);
        assert_eq!(transposed.get(1, 0), 2.0);
        assert_eq!(transposed.get(1, 1), 3.0);
        assert_eq!(transposed.get(1, 2), 6.0);
    }

    #[test]
    fn it_gets_and_sets_values() {
        let mut m = Matrix::new(2, 3);
        m.set(0, 0, 1.0);
        m.set(1, 2, 2.0);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 2.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn it_panics_on_out_of_bounds_get() {
        let m = Matrix::new(2, 3);
        m.get(2, 0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn it_panics_on_out_of_bounds_set() {
        let mut m = Matrix::new(2, 3);
        m.set(2, 0, 1.0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn it_panics_on_out_of_bounds_get_mut() {
        let mut m = Matrix::new(2, 3);
        m.get_mut(2, 0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn it_panics_on_out_of_bounds_set_mut() {
        let mut m = Matrix::new(2, 3);
        m.get_mut(2, 0);
    }

    #[test]
    fn it_gets_and_sets_mutable_values() {
        let mut m = Matrix::new(2, 3);
        *m.get_mut(0, 0) = 1.0;
        *m.get_mut(1, 2) = 2.0;
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 2.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
    }

    #[test]
    fn it_returns_mutable_data() {
        let mut m = Matrix::new(2, 3);
        m.data_mut()[0] = 1.0;
        m.data_mut()[1 * 3 + 2] = 2.0;
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 2.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
    }

    #[test]
    fn it_adds_matrices() {
        let m1 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = m1 + &m2;
        assert_eq!(result.get(0, 0), 6.0);
        assert_eq!(result.get(0, 1), 8.0);
        assert_eq!(result.get(1, 0), 10.0);
        assert_eq!(result.get(1, 1), 12.0);
    }

    #[test]
    fn it_adds_and_assigns() {
        let mut m1 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        m1 += &m2;
        assert_eq!(m1.get(0, 0), 6.0);
        assert_eq!(m1.get(0, 1), 8.0);
        assert_eq!(m1.get(1, 0), 10.0);
        assert_eq!(m1.get(1, 1), 12.0);
    }

    #[test]
    fn it_multiplies_by_scalar() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = m * 2.0;
        assert_eq!(result.get(0, 0), 2.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 6.0);
        assert_eq!(result.get(1, 1), 8.0);
    }

    #[test]
    fn it_multiplies_by_scalar_in_place() {
        let mut m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        m *= 2.0;
        assert_eq!(m.get(0, 0), 2.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 8.0);
    }

    #[test]
    fn it_multiplies_matrices() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let n = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let e = Matrix::from_vec(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        let r = &m * &n;
        assert_eq!(r, e);
    }

    #[test]
    fn it_maps() {
        let mut m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        m.apply(|x| x * 2.0);
        assert_eq!(m.get(0, 0), 2.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 8.0);
    }
}
