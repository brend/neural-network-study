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

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum ActivationFunction {
    #[default]
    Sigmoid,
    Tanh,
    Linear,
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
    InvalidLayerCount { got: usize },
    InvalidLayerSize { layer_index: usize, size: usize },
    InputLengthMismatch { expected: usize, got: usize },
    TargetLengthMismatch { expected: usize, got: usize },
}

impl fmt::Display for NeuralNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralNetworkError::InvalidLayerCount { got } => {
                write!(
                    f,
                    "invalid layer count: expected at least 2 layers (input and output), got {got}"
                )
            }
            NeuralNetworkError::InvalidLayerSize { layer_index, size } => {
                write!(
                    f,
                    "invalid layer size at index {layer_index}: expected a positive size, got {size}"
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

/// A simple feedforward neural network with an arbitrary number of layers.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layer_sizes: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    learning_rate: f64,
    activation_function: ActivationFunction,
}

impl NeuralNetwork {
    fn input_size(&self) -> usize {
        self.layer_sizes.first().copied().unwrap_or(0)
    }

    fn output_size(&self) -> usize {
        self.layer_sizes.last().copied().unwrap_or(0)
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

    /// Creates a new neural network from a layer-size specification.
    ///
    /// `layer_sizes` must include at least input and output layers,
    /// e.g. `[2, 4, 1]` or `[3, 2]` for a perceptron.
    /// Weights use Xavier-style initialization and biases start at zero.
    pub fn new(
        layer_sizes: Vec<usize>,
        rng: Option<&mut StdRng>,
    ) -> Result<Self, NeuralNetworkError> {
        if layer_sizes.len() < 2 {
            return Err(NeuralNetworkError::InvalidLayerCount {
                got: layer_sizes.len(),
            });
        }

        for (layer_index, &size) in layer_sizes.iter().enumerate() {
            if size == 0 {
                return Err(NeuralNetworkError::InvalidLayerSize { layer_index, size });
            }
        }

        let rng = match rng {
            Some(rng) => rng,
            None => &mut StdRng::from_os_rng(),
        };

        let mut weights = Vec::with_capacity(layer_sizes.len() - 1);
        let mut biases = Vec::with_capacity(layer_sizes.len() - 1);

        for pair in layer_sizes.windows(2) {
            let fan_in = pair[0];
            let fan_out = pair[1];
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();

            weights.push(Matrix::random_range(rng, fan_out, fan_in, -limit, limit));
            biases.push(Matrix::new(fan_out, 1));
        }

        Ok(NeuralNetwork {
            layer_sizes,
            weights,
            biases,
            learning_rate: 0.01,
            activation_function: ActivationFunction::default(),
        })
    }

    /// Returns the layer sizes used by this network.
    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
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

        let mut activation = Matrix::from_col_vec(input);

        for (weights, biases) in self.weights.iter().zip(self.biases.iter()) {
            let mut layer_input = weights * &activation;
            layer_input += biases;
            self.activation_function.apply(&mut layer_input);
            activation = layer_input;
        }

        Ok(activation.col(0))
    }

    /// Trains the neural network using the given input and target output.
    pub fn train(&mut self, input: Vec<f64>, target: Vec<f64>) -> Result<(), NeuralNetworkError> {
        self.validate_input_len(input.len())?;
        self.validate_target_len(target.len())?;

        let mut activations = Vec::with_capacity(self.layer_sizes.len());
        let mut activation = Matrix::from_col_vec(input);
        activations.push(activation.clone());

        for (weights, biases) in self.weights.iter().zip(self.biases.iter()) {
            let mut layer_input = weights * &activation;
            layer_input += biases;
            self.activation_function.apply(&mut layer_input);
            activation = layer_input;
            activations.push(activation.clone());
        }

        let target = Matrix::from_col_vec(target);
        let mut errors = target;
        errors -= activations
            .last()
            .expect("output layer activation is missing");

        for layer_idx in (0..self.weights.len()).rev() {
            let mut gradients = activations[layer_idx + 1].clone();
            self.activation_function.derivative(&mut gradients);
            gradients.hadamard_product(&errors);
            gradients *= self.learning_rate;

            let prev_activation_t = activations[layer_idx].transpose();
            let weight_deltas = &gradients * &prev_activation_t;

            let weights_transposed = self.weights[layer_idx].transpose();
            let next_errors = &weights_transposed * &errors;

            self.weights[layer_idx] += &weight_deltas;
            self.biases[layer_idx] += &gradients;

            errors = next_errors;
        }

        Ok(())
    }

    pub fn mutate(&mut self, rng: &mut StdRng, mutation_rate: f64) {
        for weight in &mut self.weights {
            for value in weight.data_mut().iter_mut() {
                if rng.random::<f64>() < mutation_rate {
                    *value = rng.random_range(-1.0..1.0);
                }
            }
        }

        for bias in &mut self.biases {
            for value in bias.data_mut().iter_mut() {
                if rng.random::<f64>() < mutation_rate {
                    *value = rng.random_range(-1.0..1.0);
                }
            }
        }
    }
}

#[cfg(test)]
pub mod nn_tests {
    use rand::{SeedableRng, rngs::StdRng};
    use serde_json;

    #[test]
    fn it_creates_a_neural_network() {
        let m = super::NeuralNetwork::new(vec![3, 5, 2], None).unwrap();

        assert_eq!(m.layer_sizes, vec![3, 5, 2]);
        assert_eq!(m.input_size(), 3);
        assert_eq!(m.output_size(), 2);

        assert_eq!(m.weights.len(), 2);
        assert_eq!(m.weights[0].rows(), 5);
        assert_eq!(m.weights[0].cols(), 3);
        assert_eq!(m.weights[1].rows(), 2);
        assert_eq!(m.weights[1].cols(), 5);

        assert_eq!(m.biases.len(), 2);
        assert_eq!(m.biases[0].rows(), 5);
        assert_eq!(m.biases[0].cols(), 1);
        assert_eq!(m.biases[1].rows(), 2);
        assert_eq!(m.biases[1].cols(), 1);
    }

    #[test]
    fn it_creates_a_deep_neural_network() {
        let m = super::NeuralNetwork::new(vec![3, 4, 4, 2], None).unwrap();

        assert_eq!(m.weights.len(), 3);
        assert_eq!(m.weights[0].rows(), 4);
        assert_eq!(m.weights[0].cols(), 3);
        assert_eq!(m.weights[1].rows(), 4);
        assert_eq!(m.weights[1].cols(), 4);
        assert_eq!(m.weights[2].rows(), 2);
        assert_eq!(m.weights[2].cols(), 4);

        assert_eq!(m.biases.len(), 3);
        assert_eq!(m.biases[0].rows(), 4);
        assert_eq!(m.biases[1].rows(), 4);
        assert_eq!(m.biases[2].rows(), 2);
    }

    #[test]
    fn it_creates_a_no_hidden_layer_network() {
        let m = super::NeuralNetwork::new(vec![3, 2], None).unwrap();

        assert_eq!(m.weights.len(), 1);
        assert_eq!(m.biases.len(), 1);
        assert_eq!(m.weights[0].rows(), 2);
        assert_eq!(m.weights[0].cols(), 3);
        assert_eq!(m.biases[0].rows(), 2);
        assert_eq!(m.biases[0].cols(), 1);
    }

    #[test]
    pub fn it_predicts() {
        let m = super::NeuralNetwork::new(vec![3, 5, 2], None).unwrap();
        let input = vec![0.5, 0.2, 0.1];
        let output = m.predict(input).unwrap();
        assert_eq!(output.len(), 2);
        assert_ne!(output[0], output[1]);
    }

    #[test]
    fn predict_handles_deep_and_no_hidden_architectures() {
        let deep = super::NeuralNetwork::new(vec![3, 4, 4, 2], None).unwrap();
        let no_hidden = super::NeuralNetwork::new(vec![3, 2], None).unwrap();

        assert_eq!(deep.predict(vec![0.1, 0.2, 0.3]).unwrap().len(), 2);
        assert_eq!(no_hidden.predict(vec![0.1, 0.2, 0.3]).unwrap().len(), 2);
    }

    #[test]
    fn predict_linear_activation_matches_manual_multilayer_math() {
        let mut nn = super::NeuralNetwork::new(vec![2, 2, 1], None).unwrap();
        nn.set_activation_function(super::ActivationFunction::Linear);

        nn.weights[0] = super::Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        nn.biases[0] = super::Matrix::from_col_vec(vec![0.5, -0.5]);
        nn.weights[1] = super::Matrix::from_vec(1, 2, vec![2.0, -1.0]);
        nn.biases[1] = super::Matrix::from_col_vec(vec![1.0]);

        let output = nn.predict(vec![0.25, 0.75]).unwrap();
        assert!((output[0] - 2.25).abs() < 1e-12);
    }

    #[test]
    fn train_updates_all_layers_in_deep_network() {
        let mut nn = super::NeuralNetwork::new(vec![2, 3, 2, 1], None).unwrap();
        nn.set_activation_function(super::ActivationFunction::Linear);
        nn.set_learning_rate(0.1);

        nn.weights[0] = super::Matrix::from_vec(3, 2, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weights[1] = super::Matrix::from_vec(2, 3, vec![0.2, 0.1, 0.4, 0.3, 0.5, 0.7]);
        nn.weights[2] = super::Matrix::from_vec(1, 2, vec![0.9, 0.8]);
        nn.biases[0] = super::Matrix::from_col_vec(vec![0.0, 0.0, 0.0]);
        nn.biases[1] = super::Matrix::from_col_vec(vec![0.0, 0.0]);
        nn.biases[2] = super::Matrix::from_col_vec(vec![0.0]);

        let weights_before: Vec<Vec<f64>> = nn.weights.iter().map(|w| w.data().clone()).collect();
        let biases_before: Vec<Vec<f64>> = nn.biases.iter().map(|b| b.data().clone()).collect();

        nn.train(vec![0.9, 0.1], vec![0.2]).unwrap();

        for (idx, weight) in nn.weights.iter().enumerate() {
            assert_ne!(
                weight.data(),
                &weights_before[idx],
                "expected weights at layer {idx} to change"
            );
        }
        for (idx, bias) in nn.biases.iter().enumerate() {
            assert_ne!(
                bias.data(),
                &biases_before[idx],
                "expected biases at layer {idx} to change"
            );
        }
    }

    #[test]
    fn mutate_honors_rate_extremes() {
        let mut rng = StdRng::seed_from_u64(77);
        let mut nn = super::NeuralNetwork::new(vec![3, 4, 2], Some(&mut rng)).unwrap();

        let original_weights: Vec<Vec<f64>> = nn.weights.iter().map(|w| w.data().clone()).collect();
        let original_biases: Vec<Vec<f64>> = nn.biases.iter().map(|b| b.data().clone()).collect();

        nn.mutate(&mut rng, 0.0);
        for (idx, weight) in nn.weights.iter().enumerate() {
            assert_eq!(weight.data(), &original_weights[idx]);
        }
        for (idx, bias) in nn.biases.iter().enumerate() {
            assert_eq!(bias.data(), &original_biases[idx]);
        }

        nn.mutate(&mut rng, 1.0);
        let mut any_changed = false;

        for (idx, weight) in nn.weights.iter().enumerate() {
            if weight.data() != &original_weights[idx] {
                any_changed = true;
            }
            assert!(
                weight
                    .data()
                    .iter()
                    .all(|value| *value >= -1.0 && *value < 1.0)
            );
        }
        for (idx, bias) in nn.biases.iter().enumerate() {
            if bias.data() != &original_biases[idx] {
                any_changed = true;
            }
            assert!(
                bias.data()
                    .iter()
                    .all(|value| *value >= -1.0 && *value < 1.0)
            );
        }

        assert!(any_changed, "expected at least one parameter to change");
    }

    #[test]
    fn it_learns_the_or_function() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut nn = super::NeuralNetwork::new(vec![2, 4, 1], Some(&mut rng)).unwrap();
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
        let nn = super::NeuralNetwork::new(vec![3, 5, 2], None).unwrap();

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
        let mut nn = super::NeuralNetwork::new(vec![3, 5, 2], None).unwrap();

        assert_eq!(
            nn.train(vec![0.1, 0.2, 0.3], vec![1.0]),
            Err(super::NeuralNetworkError::TargetLengthMismatch {
                expected: 2,
                got: 1,
            })
        );
    }

    #[test]
    fn new_rejects_invalid_layer_vectors() {
        assert_eq!(
            super::NeuralNetwork::new(vec![], None).unwrap_err(),
            super::NeuralNetworkError::InvalidLayerCount { got: 0 }
        );

        assert_eq!(
            super::NeuralNetwork::new(vec![3], None).unwrap_err(),
            super::NeuralNetworkError::InvalidLayerCount { got: 1 }
        );

        assert_eq!(
            super::NeuralNetwork::new(vec![0, 5, 2], None).unwrap_err(),
            super::NeuralNetworkError::InvalidLayerSize {
                layer_index: 0,
                size: 0,
            }
        );

        assert_eq!(
            super::NeuralNetwork::new(vec![3, 0, 2], None).unwrap_err(),
            super::NeuralNetworkError::InvalidLayerSize {
                layer_index: 1,
                size: 0,
            }
        );

        assert_eq!(
            super::NeuralNetwork::new(vec![3, 5, 0], None).unwrap_err(),
            super::NeuralNetworkError::InvalidLayerSize {
                layer_index: 2,
                size: 0,
            }
        );
    }

    #[test]
    fn new_uses_zero_biases() {
        let nn = super::NeuralNetwork::new(vec![3, 5, 4, 2], None).unwrap();

        assert!(
            nn.biases
                .iter()
                .all(|bias| bias.data().iter().all(|value| *value == 0.0))
        );
    }

    #[test]
    fn new_uses_xavier_weight_ranges() {
        let mut rng = StdRng::seed_from_u64(7);
        let layer_sizes = vec![3, 5, 4, 2];
        let nn = super::NeuralNetwork::new(layer_sizes.clone(), Some(&mut rng)).unwrap();

        for (weight, pair) in nn.weights.iter().zip(layer_sizes.windows(2)) {
            let fan_in = pair[0] as f64;
            let fan_out = pair[1] as f64;
            let limit = (6.0_f64 / (fan_in + fan_out)).sqrt();

            assert!(
                weight
                    .data()
                    .iter()
                    .all(|value| *value >= -limit && *value < limit)
            );
        }
    }

    #[test]
    fn it_learns_the_xor_function() {
        let mut rng = StdRng::seed_from_u64(99);
        let mut nn = super::NeuralNetwork::new(vec![2, 4, 1], Some(&mut rng)).unwrap();
        nn.set_learning_rate(0.5);

        let training_data = [
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        for _ in 0..20_000 {
            for (input, target) in &training_data {
                nn.train(input.clone(), target.clone()).unwrap();
            }
        }

        assert!(nn.predict(vec![0.0, 0.0]).unwrap()[0] < 0.2);
        assert!(nn.predict(vec![0.0, 1.0]).unwrap()[0] > 0.8);
        assert!(nn.predict(vec![1.0, 0.0]).unwrap()[0] > 0.8);
        assert!(nn.predict(vec![1.0, 1.0]).unwrap()[0] < 0.2);
    }

    #[test]
    fn it_learns_the_xor_function_with_deeper_network() {
        let mut rng = StdRng::seed_from_u64(314);
        let mut nn = super::NeuralNetwork::new(vec![2, 4, 4, 1], Some(&mut rng)).unwrap();
        nn.set_learning_rate(0.5);

        let training_data = [
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        for _ in 0..20_000 {
            for (input, target) in &training_data {
                nn.train(input.clone(), target.clone()).unwrap();
            }
        }

        assert!(nn.predict(vec![0.0, 0.0]).unwrap()[0] < 0.2);
        assert!(nn.predict(vec![0.0, 1.0]).unwrap()[0] > 0.8);
        assert!(nn.predict(vec![1.0, 0.0]).unwrap()[0] > 0.8);
        assert!(nn.predict(vec![1.0, 1.0]).unwrap()[0] < 0.2);
    }

    #[test]
    fn perceptron_architecture_learns_linearly_separable_data() {
        let mut rng = StdRng::seed_from_u64(202);
        let mut nn = super::NeuralNetwork::new(vec![2, 1], Some(&mut rng)).unwrap();
        nn.set_learning_rate(0.5);

        let training_data = [
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];

        for _ in 0..12_000 {
            for (input, target) in &training_data {
                nn.train(input.clone(), target.clone()).unwrap();
            }
        }

        assert!(nn.predict(vec![0.0, 0.0]).unwrap()[0] < 0.2);
        assert!(nn.predict(vec![0.0, 1.0]).unwrap()[0] < 0.2);
        assert!(nn.predict(vec![1.0, 0.0]).unwrap()[0] > 0.8);
        assert!(nn.predict(vec![1.0, 1.0]).unwrap()[0] > 0.8);
    }

    #[test]
    fn serde_round_trip_preserves_predictions() {
        let mut rng = StdRng::seed_from_u64(123);
        let mut nn = super::NeuralNetwork::new(vec![2, 4, 1], Some(&mut rng)).unwrap();
        nn.set_learning_rate(0.5);

        let training_data = [
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        for _ in 0..5_000 {
            for (input, target) in &training_data {
                nn.train(input.clone(), target.clone()).unwrap();
            }
        }

        let probe_input = vec![0.25, 0.75];
        let before = nn.predict(probe_input.clone()).unwrap();

        let json = serde_json::to_string(&nn).unwrap();
        let restored: super::NeuralNetwork = serde_json::from_str(&json).unwrap();
        let after = restored.predict(probe_input).unwrap();

        assert_eq!(before, after);
    }
}
