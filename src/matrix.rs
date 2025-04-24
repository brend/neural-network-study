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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data.len(), 2);
        assert_eq!(m.data[0].len(), 3);
        assert_eq!(m.data[1].len(), 3);
        assert_eq!(m.data[0][0], 0.0);
        assert_eq!(m.data[0][1], 0.0);
        assert_eq!(m.data[0][2], 0.0);
        assert_eq!(m.data[1][0], 0.0);
        assert_eq!(m.data[1][1], 0.0);
        assert_eq!(m.data[1][2], 0.0);
    }
}