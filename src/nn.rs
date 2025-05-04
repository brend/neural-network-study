use crate::matrix::Matrix;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

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
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        rng: Option<&mut StdRng>,
    ) -> Self {
        let rng = match rng {
            Some(rng) => rng,
            None => &mut StdRng::from_os_rng(),
        };
        NeuralNetwork {
            weights_input_hidden: Matrix::random(rng, hidden_size, input_size),
            weights_hidden_output: Matrix::random(rng, output_size, hidden_size),
            biases_hidden: Matrix::random(rng, hidden_size, 1),
            biases_output: Matrix::random(rng, output_size, 1),
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
        let output_layer_input =
            &self.weights_hidden_output * &hidden_layer_output + &self.biases_output;
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
        let output_layer_input =
            &self.weights_hidden_output * &hidden_layer_output + &self.biases_output;
        let mut output_layer_output = output_layer_input;
        self.activation_function.apply(&mut output_layer_output);

        // Create target matrix
        let target = Matrix::from_col_vec(target);

        // Calculate the error
        // ERROR = TARGET - OUTPUT
        let mut output_errors = target;
        output_errors -= &output_layer_output;

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

    pub fn mutate(&mut self, rng: &mut StdRng, mutation_rate: f64) {
        for i in 0..self.weights_input_hidden.rows() {
            for j in 0..self.weights_input_hidden.cols() {
                if rng.random::<f64>() < mutation_rate {
                    self.weights_input_hidden
                        .set(i, j, rng.random_range(-1.0..1.0));
                }
            }
        }
        for i in 0..self.weights_hidden_output.rows() {
            for j in 0..self.weights_hidden_output.cols() {
                if rng.random::<f64>() < mutation_rate {
                    self.weights_hidden_output
                        .set(i, j, rng.random_range(-1.0..1.0));
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
        let m = super::NeuralNetwork::new(3, 5, 2, None);
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
        let m = super::NeuralNetwork::new(3, 5, 2, None);
        let input = vec![0.5, 0.2, 0.1];
        let output = m.predict(input.clone());
        assert_eq!(output.len(), 2);
        assert_ne!(output[0], output[1]);
    }
}
