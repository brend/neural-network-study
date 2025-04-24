/// A simple 2-dimensional matrix with basic operations
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    /// Creates a new matrix with the given number of rows and columns,
    /// initialized to zero.
    fn new(rows: usize, cols: usize) -> Self {
        let data = vec![vec![0.0; cols]; rows];
        Self { rows, cols, data }
    }

    /// Creates a new matrix with the given number of rows and columns,
    /// initialized with random values between -1.0 and 1.0.
    fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..rows)
            .map(|_| (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        Self { rows, cols, data }
    }

    /// Transposes the matrix.
    fn transpose(&self) -> Self {
        let mut transposed_data = vec![vec![0.0; self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed_data[j][i] = self.data[i][j];
            }
        }   
        Self::from_vec(transposed_data)
    }
}