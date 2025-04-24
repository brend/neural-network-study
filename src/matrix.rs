use rand::Rng;
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// A simple 2-dimensional matrix with basic operations
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    /// Creates a new matrix with the given number of rows and columns,
    /// initialized to zero.
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![vec![0.0; cols]; rows];
        Self { rows, cols, data }
    }

    /// Creates a new matrix with the given number of rows and columns,
    /// initialized with random values between -1.0 and 1.0.
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::r#rng();
        let data = (0..rows)
            .map(|_| (0..cols).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        Self { rows, cols, data }
    }

    /// Creates a new matrix from a vector of vectors.
    /// The outer vector represents the rows,
    /// and the inner vectors represent the columns.
    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { rows, cols, data }
    }

    /// Transposes the matrix.
    pub fn transpose(&self) -> Self {
        let mut transposed_data = vec![vec![0.0; self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed_data[j][i] = self.data[i][j];
            }
        }   
        Self::from_vec(transposed_data)
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns a reference to the data in the matrix.
    pub fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    /// Returns a mutable reference to the data in the matrix.
    pub fn data_mut(&mut self) -> &mut Vec<Vec<f64>> {
        &mut self.data
    }

    /// Returns the value at the given row and column.
    /// Panics if the indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        self.data[row][col]
    }

    /// Returns a mutable reference to the value at the given row and column.
    /// Panics if the indices are out of bounds.
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        &mut self.data[row][col]
    }

    /// Sets the value at the given row and column.
    /// Panics if the indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        self.data[row][col] = value;
    }

    /// Returns the matrix resulting from 
    /// applying the function `f` to each element of the matrix.
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, f(self.get(i, j)));
            }
        }
        result
    }

    /// Applies the function `f` to each element of the matrix in place.
    /// This is an in-place operation.
    pub fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, f(self.get(i, j)));
            }
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

impl Mul<f64> for Matrix {
    type Output = Matrix;

    /// Multiplies the matrix by a scalar.
    fn mul(self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) * scalar);
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
                self.set(i, j, self.get(i, j) * scalar);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data().len(), 2);
        assert_eq!(m.data[0].len(), 3);
        assert_eq!(m.data[1].len(), 3);
        assert_eq!(m.data[0][0], 0.0);
        assert_eq!(m.data[0][1], 0.0);
        assert_eq!(m.data[0][2], 0.0);
        assert_eq!(m.data[1][0], 0.0);
        assert_eq!(m.data[1][1], 0.0);
        assert_eq!(m.data[1][2], 0.0);
    }

    #[test]
    fn it_creates_random_matrix() {
        let m = Matrix::random(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data.len(), 2);
        assert_eq!(m.data[0].len(), 3);
        assert_eq!(m.data[1].len(), 3);
        for i in 0..2 {
            for j in 0..3 {
                assert!(m.data[i][j] >= -1.0 && m.data[i][j] <= 1.0);
            }
        }
    }

    #[test]
    fn it_creates_a_matrix_from_a_vector() {
        let v = vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]];
        let m = Matrix::from_vec(v.clone());
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data, v);
    }

    #[test]
    fn it_transposes_matrix() {
        let m = Matrix::from_vec(vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]]);
        let transposed = m.transpose();
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 2);
        assert_eq!(transposed.data[0][0], 1.0);
        assert_eq!(transposed.data[0][1], 3.0);
        assert_eq!(transposed.data[1][0], 2.0);
        assert_eq!(transposed.data[1][1], 4.0);
        assert_eq!(transposed.data[2][0], 5.0);
        assert_eq!(transposed.data[2][1], 6.0);
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
        m.data_mut()[0][0] = 1.0;
        m.data_mut()[1][2] = 2.0;
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 2.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
    }

    #[test]
    fn it_adds_matrices() {
        let m1 = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2 = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let result = m1 + &m2;
        assert_eq!(result.get(0, 0), 6.0);
        assert_eq!(result.get(0, 1), 8.0);
        assert_eq!(result.get(1, 0), 10.0);
        assert_eq!(result.get(1, 1), 12.0);
    }

    #[test]
    fn it_adds_and_assigns() {
        let mut m1 = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2 = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        m1 += &m2;
        assert_eq!(m1.get(0, 0), 6.0);
        assert_eq!(m1.get(0, 1), 8.0);
        assert_eq!(m1.get(1, 0), 10.0);
        assert_eq!(m1.get(1, 1), 12.0);
    }

    #[test]
    fn it_multiplies_by_scalar() {
        let m = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = m * 2.0;
        assert_eq!(result.get(0, 0), 2.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 6.0);
        assert_eq!(result.get(1, 1), 8.0);
    }

    #[test]
    fn it_multiplies_by_scalar_in_place() {
        let mut m = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        m *= 2.0;
        assert_eq!(m.get(0, 0), 2.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 8.0);
    }

    #[test]
    fn it_maps() {
        let m = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = m.map(|x| x * 2.0);
        assert_eq!(result.get(0, 0), 2.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 6.0);
        assert_eq!(result.get(1, 1), 8.0);
    }

    #[test]
    fn it_maps_mut() {
        let mut m = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        m.map_mut(|x| x * 2.0);
        assert_eq!(m.get(0, 0), 2.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 8.0);
    }
}