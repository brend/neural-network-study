use rand::Rng;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};

/// A simple 2-dimensional matrix with basic operations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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

    /// Creates a new matrix from a 2D vector.
    /// The outer vector represents the rows, and the inner vectors represent the columns.
    /// Panics if the inner vectors have different lengths.
    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        for row in &data {
            if row.len() != cols {
                panic!("All rows must have the same number of columns");
            }
        }
        Self { rows, cols, data }
    }

    /// Creates a new matrix from a column vector.
    pub fn from_col_vec(data: Vec<f64>) -> Self {
        let rows = data.len();
        let cols = 1;
        let data = data.into_iter().map(|x| vec![x]).collect();
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

    /// Returns the column at the given index as a vector.
    /// Panics if the index is out of bounds.
    pub fn col(&self, index: usize) -> Vec<f64> {
        if index >= self.cols {
            panic!("Index out of bounds");
        }
        (0..self.rows).map(|i| self.data[i][index]).collect()
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

    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = f(self.data[i][j]);
            }
        }
    }

    pub fn hadamar_product(&mut self, other: &Matrix) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrices must have the same dimensions for Hadamard product");
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, self.get(i, j) * other.get(i, j));
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

impl Sub<&Matrix> for Matrix  {
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

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    /// Multiplies two matrices together.
    /// Panics if the matrices have incompatible dimensions.
    fn mul(self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Matrices have incompatible dimensions for multiplication");
        }
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }
}

#[cfg(test)]
mod matrix_tests {
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
        let mut m = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        m.apply(|x| x * 2.0);
        assert_eq!(m.get(0, 0), 2.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 8.0);
    }
}

fn sigmoid(x: &mut Matrix) {
    x.apply(|x| 1.0 / (1.0 + (-x).exp()))
}

fn sigmoid_derivative(x: &mut Matrix) {
    x.apply(|x| x * (1.0 - x))
}

fn tanh(x: &mut Matrix) {
    x.apply(|x| x.tanh())
}

fn tanh_derivative(x: &mut Matrix) {
    x.apply(|x| 1.0 - x.tanh().powi(2))
}

fn linear(_: &mut Matrix) {}

fn linear_derivative(x: &mut Matrix) {
    x.apply(|_| 1.0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Linear,
}

impl Default for ActivationFunction {
    fn default() -> Self {
        ActivationFunction::Sigmoid
    }
}

impl ActivationFunction {
    fn apply(&self, x: &mut Matrix) {
        match self {
            ActivationFunction::Sigmoid => sigmoid(x),
            ActivationFunction::Tanh => tanh(x),
            ActivationFunction::Linear => linear(x),
        }
    }

    fn derivative(&self, x: &mut Matrix) {
        match self {
            ActivationFunction::Sigmoid => sigmoid_derivative(x),
            ActivationFunction::Tanh => tanh_derivative(x),
            ActivationFunction::Linear => linear_derivative(x),
        }
    }
}

/// A simple feedforward neural network with one hidden layer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NeuralNetwork {
    weights_input_hidden: Matrix,
    weights_hidden_output: Matrix,
    biases_hidden: Matrix,
    biases_output: Matrix,
    learning_rate: f64,
    activation_function: ActivationFunction,
}

impl NeuralNetwork {
    /// Creates a new neural network with the given sizes for input, hidden, and output layers.
    /// The weights and biases are initialized randomly.
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        NeuralNetwork {
            weights_input_hidden: Matrix::random(hidden_size, input_size),
            weights_hidden_output: Matrix::random(output_size, hidden_size),
            biases_hidden: Matrix::random(hidden_size, 1),
            biases_output: Matrix::random(output_size, 1),
            learning_rate: 0.01,
            activation_function: ActivationFunction::default(),
        }
    }

    /// Returns the learning rate of the neural network.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Sets the learning rate for the neural network.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    /// Returns the activation function of the neural network.
    pub fn activation_function(&self) -> &ActivationFunction {
        &self.activation_function
    }

    /// Sets the activation function for the neural network.
    pub fn set_activation_function(&mut self, activation_function: ActivationFunction) {
        self.activation_function = activation_function;
    }

    /// Predicts the output for the given input using the neural network.
    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        // Generate the hidden outputs
        let input_matrix = Matrix::from_col_vec(input);
        let mut hidden_layer_input = &self.weights_input_hidden * &input_matrix;
        hidden_layer_input += &self.biases_hidden;
        let mut hidden_layer_output = hidden_layer_input;
        self.activation_function.apply(&mut hidden_layer_output);
        // Generate the output's output
        let output_layer_input = &self.weights_hidden_output * &hidden_layer_output + &self.biases_output;
        let mut output_layer_output = output_layer_input;
        self.activation_function.apply(&mut output_layer_output);
        // Return the output as a vector
        output_layer_output.col(0)
    }

    /// Trains the neural network using the given input and target output.
    /// The input and target should be vectors of the same length as the input and output sizes of the network.
    /// The training process involves forward propagation and backpropagation to adjust the weights and biases.
    pub fn train(&mut self, input: Vec<f64>, target: Vec<f64>) {
        // Generate the hidden outputs
        let input_matrix = Matrix::from_col_vec(input);
        let mut hidden_layer_input = &self.weights_input_hidden * &input_matrix;
        hidden_layer_input += &self.biases_hidden;
        let mut hidden_layer_output = hidden_layer_input;
        self.activation_function.apply(&mut hidden_layer_output);
        // Generate the output's outputs
        let output_layer_input = &self.weights_hidden_output * &hidden_layer_output + &self.biases_output;
        let mut output_layer_output = output_layer_input;
        self.activation_function.apply(&mut output_layer_output);
        
        // Create target matrix
        let target = Matrix::from_col_vec(target);

        // Calculate the error
        // ERROR = TARGET - OUTPUT
        let output_errors = target - &output_layer_output;

        // Calculate gradients
        let mut gradients = output_layer_output;
        self.activation_function.derivative(&mut gradients);
        gradients.hadamar_product(&output_errors);
        gradients *= self.learning_rate;

        // Calculcate deltas
        let hidden_transposed = hidden_layer_output.transpose();
        let weight_hidden_output_deltas = &gradients * &hidden_transposed;

        // Adjust the weights by deltas
        self.weights_hidden_output += &weight_hidden_output_deltas;
        // Adjust the bias by its deltas (which is just the gradients)
        self.biases_output += &gradients;

        // Calculate the hidden layer errors
        let weight_hidden_output_transposed = self.weights_hidden_output.transpose();
        let hidden_errors = &weight_hidden_output_transposed * &output_errors;

        // Calculate hidden gradients
        let mut hidden_gradient = hidden_layer_output;
        self.activation_function.derivative(&mut hidden_gradient);
        hidden_gradient.hadamar_product(&hidden_errors);
        hidden_gradient *= self.learning_rate;

        // Calculate input -> hidden deltas
        let inputs_transposed = input_matrix.transpose();
        let weight_input_hidden_deltas = &hidden_gradient * &inputs_transposed;

        self.weights_input_hidden += &weight_input_hidden_deltas;
        // Adjust the bias by its deltas (which is just the gradient)
        self.biases_hidden += &hidden_gradient;
    }

    pub fn mutate(&mut self, mutation_rate: f64) {
        let mut rng = rand::rng();
        for i in 0..self.weights_input_hidden.rows() {
            for j in 0..self.weights_input_hidden.cols() {
                if rng.random::<f64>() < mutation_rate {
                    self.weights_input_hidden.set(i, j, rng.random_range(-1.0..1.0));
                }
            }
        }
        for i in 0..self.weights_hidden_output.rows() {
            for j in 0..self.weights_hidden_output.cols() {
                if rng.random::<f64>() < mutation_rate {
                    self.weights_hidden_output.set(i, j, rng.random_range(-1.0..1.0));
                }
            }
        }
        for i in 0..self.biases_hidden.rows() {
            if rng.random::<f64>() < mutation_rate {
                self.biases_hidden.set(i, 0, rng.random_range(-1.0..1.0));
            }
        }
        for i in 0..self.biases_output.rows() {
            if rng.random::<f64>() < mutation_rate {
                self.biases_output.set(i, 0, rng.random_range(-1.0..1.0));
            }
        }
    }
}

pub mod nn_tests {
    #[test]
    fn it_creates_a_neural_network() {
        let m = super::NeuralNetwork::new(3, 5, 2);
        assert_eq!(m.weights_input_hidden.rows(), 5);
        assert_eq!(m.weights_input_hidden.cols(), 3);
        assert_eq!(m.weights_hidden_output.rows(), 2);
        assert_eq!(m.weights_hidden_output.cols(), 5);
        assert_eq!(m.biases_hidden.rows(), 5);
        assert_eq!(m.biases_hidden.cols(), 1);
        assert_eq!(m.biases_output.rows(), 2);
        assert_eq!(m.biases_output.cols(), 1);
    }

    #[test]
    pub fn it_predicts() {
        let m = super::NeuralNetwork::new(3, 5, 2);
        let input = vec![0.5, 0.2, 0.1];
        let output = m.predict(input.clone());
        assert_eq!(output.len(), 2);
        assert_ne!(output[0], output[1]);
    }
}
