use crate::matrix::Matrix;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};

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
    x.apply(|x| 1.0 - x.powi(2))
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NeuralNetworkError {
    InvalidLayerSize {
        layer: &'static str,
        size: usize,
    },
    InputLengthMismatch {
        expected: usize,
        got: usize,
    },
    TargetLengthMismatch {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for NeuralNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralNetworkError::InvalidLayerSize { layer, size } => {
                write!(
                    f,
                    "invalid {layer} layer size: expected a positive size, got {size}"
                )
            }
            NeuralNetworkError::InputLengthMismatch { expected, got } => {
                write!(f, "input length mismatch: expected {expected}, got {got}")
            }
            NeuralNetworkError::TargetLengthMismatch { expected, got } => {
                write!(f, "target length mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl Error for NeuralNetworkError {}

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
    fn input_size(&self) -> usize {
        self.weights_input_hidden.cols()
    }

    fn output_size(&self) -> usize {
        self.weights_hidden_output.rows()
    }

    fn validate_input_len(&self, actual: usize) -> Result<(), NeuralNetworkError> {
        if actual == self.input_size() {
            Ok(())
        } else {
            Err(NeuralNetworkError::InputLengthMismatch {
                expected: self.input_size(),
                got: actual,
            })
        }
    }

    fn validate_target_len(&self, actual: usize) -> Result<(), NeuralNetworkError> {
        if actual == self.output_size() {
            Ok(())
        } else {
            Err(NeuralNetworkError::TargetLengthMismatch {
                expected: self.output_size(),
                got: actual,
            })
        }
    }

    /// Creates a new neural network with the given sizes for input, hidden, and output layers.
    /// The weights use Xavier-style initialization and biases start at zero.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        rng: Option<&mut StdRng>,
    ) -> Result<Self, NeuralNetworkError> {
        if input_size == 0 {
            return Err(NeuralNetworkError::InvalidLayerSize {
                layer: "input",
                size: input_size,
            });
        }
        if hidden_size == 0 {
            return Err(NeuralNetworkError::InvalidLayerSize {
                layer: "hidden",
                size: hidden_size,
            });
        }
        if output_size == 0 {
            return Err(NeuralNetworkError::InvalidLayerSize {
                layer: "output",
                size: output_size,
            });
        }

        let rng = match rng {
            Some(rng) => rng,
            None => &mut StdRng::from_os_rng(),
        };

        let limit_input_hidden = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let limit_hidden_output = (6.0 / (hidden_size + output_size) as f64).sqrt();

        Ok(NeuralNetwork {
            weights_input_hidden: Matrix::random_range(
                rng,
                hidden_size,
                input_size,
                -limit_input_hidden,
                limit_input_hidden,
            ),
            weights_hidden_output: Matrix::random_range(
                rng,
                output_size,
                hidden_size,
                -limit_hidden_output,
                limit_hidden_output,
            ),
            biases_hidden: Matrix::new(hidden_size, 1),
            biases_output: Matrix::new(output_size, 1),
            learning_rate: 0.01,
            activation_function: ActivationFunction::default(),
        })
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
    pub fn predict(&self, input: Vec<f64>) -> Result<Vec<f64>, NeuralNetworkError> {
        self.validate_input_len(input.len())?;

        let input_matrix = Matrix::from_col_vec(input);
        let mut hidden_layer_input = &self.weights_input_hidden * &input_matrix;
        hidden_layer_input += &self.biases_hidden;
        let mut hidden_layer_output = hidden_layer_input;
        self.activation_function.apply(&mut hidden_layer_output);

        let output_layer_input =
            &self.weights_hidden_output * &hidden_layer_output + &self.biases_output;
        let mut output_layer_output = output_layer_input;
        self.activation_function.apply(&mut output_layer_output);

        Ok(output_layer_output.col(0))
    }

    /// Trains the neural network using the given input and target output.
    pub fn train(
        &mut self,
        input: Vec<f64>,
        target: Vec<f64>,
    ) -> Result<(), NeuralNetworkError> {
        self.validate_input_len(input.len())?;
        self.validate_target_len(target.len())?;

        let input_matrix = Matrix::from_col_vec(input);
        let mut hidden_layer_input = &self.weights_input_hidden * &input_matrix;
        hidden_layer_input += &self.biases_hidden;
        let mut hidden_layer_output = hidden_layer_input;
        self.activation_function.apply(&mut hidden_layer_output);

        let output_layer_input =
            &self.weights_hidden_output * &hidden_layer_output + &self.biases_output;
        let mut output_layer_output = output_layer_input;
        self.activation_function.apply(&mut output_layer_output);

        let target = Matrix::from_col_vec(target);

        let mut output_errors = target;
        output_errors -= &output_layer_output;

        let mut gradients = output_layer_output;
        self.activation_function.derivative(&mut gradients);
        gradients.hadamard_product(&output_errors);
        gradients *= self.learning_rate;

        let hidden_transposed = hidden_layer_output.transpose();
        let weight_hidden_output_deltas = &gradients * &hidden_transposed;

        let weight_hidden_output_transposed = self.weights_hidden_output.transpose();
        let hidden_errors = &weight_hidden_output_transposed * &output_errors;

        self.weights_hidden_output += &weight_hidden_output_deltas;
        self.biases_output += &gradients;

        let mut hidden_gradient = hidden_layer_output;
        self.activation_function.derivative(&mut hidden_gradient);
        hidden_gradient.hadamard_product(&hidden_errors);
        hidden_gradient *= self.learning_rate;

        let inputs_transposed = input_matrix.transpose();
        let weight_input_hidden_deltas = &hidden_gradient * &inputs_transposed;
        self.weights_input_hidden += &weight_input_hidden_deltas;
        self.biases_hidden += &hidden_gradient;

        Ok(())
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

#[cfg(test)]
pub mod nn_tests {
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn it_creates_a_neural_network() {
        let m = super::NeuralNetwork::new(3, 5, 2, None).unwrap();
        assert_eq!(m.weights_input_hidden.rows(), 5);
        assert_eq!(m.input_size(), 3);
        assert_eq!(m.output_size(), 2);
        assert_eq!(m.weights_hidden_output.cols(), 5);
        assert_eq!(m.biases_hidden.rows(), 5);
        assert_eq!(m.biases_hidden.cols(), 1);
        assert_eq!(m.biases_output.rows(), 2);
        assert_eq!(m.biases_output.cols(), 1);
    }

    #[test]
    pub fn it_predicts() {
        let m = super::NeuralNetwork::new(3, 5, 2, None).unwrap();
        let input = vec![0.5, 0.2, 0.1];
        let output = m.predict(input).unwrap();
        assert_eq!(output.len(), 2);
        assert_ne!(output[0], output[1]);
    }

    #[test]
    fn it_learns_the_or_function() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut nn = super::NeuralNetwork::new(2, 4, 1, Some(&mut rng)).unwrap();
        nn.set_learning_rate(0.5);

        let training_data = [
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];

        for _ in 0..10_000 {
            for (input, target) in &training_data {
                nn.train(input.clone(), target.clone()).unwrap();
            }
        }

        assert!(nn.predict(vec![0.0, 0.0]).unwrap()[0] < 0.2);
        assert!(nn.predict(vec![0.0, 1.0]).unwrap()[0] > 0.8);
        assert!(nn.predict(vec![1.0, 0.0]).unwrap()[0] > 0.8);
        assert!(nn.predict(vec![1.0, 1.0]).unwrap()[0] > 0.8);
    }

    #[test]
    fn tanh_derivative_uses_activated_output() {
        let mut x = crate::Matrix::from_col_vec(vec![0.5, -0.25]);
        super::tanh_derivative(&mut x);

        assert!((x.get(0, 0) - 0.75).abs() < 1e-12);
        assert!((x.get(1, 0) - 0.9375).abs() < 1e-12);
    }

    #[test]
    fn predict_returns_clear_error_for_wrong_input_size() {
        let nn = super::NeuralNetwork::new(3, 5, 2, None).unwrap();

        assert_eq!(
            nn.predict(vec![0.1, 0.2]),
            Err(super::NeuralNetworkError::InputLengthMismatch {
                expected: 3,
                got: 2,
            })
        );
    }

    #[test]
    fn train_returns_clear_error_for_wrong_target_size() {
        let mut nn = super::NeuralNetwork::new(3, 5, 2, None).unwrap();

        assert_eq!(
            nn.train(vec![0.1, 0.2, 0.3], vec![1.0]),
            Err(super::NeuralNetworkError::TargetLengthMismatch {
                expected: 2,
                got: 1,
            })
        );
    }

    #[test]
    fn new_rejects_zero_sized_layers() {
        assert_eq!(
            super::NeuralNetwork::new(0, 5, 2, None).unwrap_err(),
            super::NeuralNetworkError::InvalidLayerSize {
                layer: "input",
                size: 0,
            }
        );
        assert_eq!(
            super::NeuralNetwork::new(3, 0, 2, None).unwrap_err(),
            super::NeuralNetworkError::InvalidLayerSize {
                layer: "hidden",
                size: 0,
            }
        );
        assert_eq!(
            super::NeuralNetwork::new(3, 5, 0, None).unwrap_err(),
            super::NeuralNetworkError::InvalidLayerSize {
                layer: "output",
                size: 0,
            }
        );
    }

    #[test]
    fn new_uses_zero_biases() {
        let nn = super::NeuralNetwork::new(3, 5, 2, None).unwrap();

        assert!(nn.biases_hidden.data().iter().all(|value| *value == 0.0));
        assert!(nn.biases_output.data().iter().all(|value| *value == 0.0));
    }
}
