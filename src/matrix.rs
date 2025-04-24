use rand::Rng;


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
}