use crate::matrix::Matrix;

fn sigmoid(x: &Matrix) -> Matrix {
    x.map(|x| 1.0 / (1.0 + (-x).exp()))
}

fn sigmoid_derivative(x: &Matrix) -> Matrix {
    x.map(|x| x * (1.0 - x))
}

fn tanh(x: &Matrix) -> Matrix {
    x.map(|x| x.tanh())
}

fn tanh_derivative(x: &Matrix) -> Matrix {
    x.map(|x| 1.0 - x.tanh().powi(2))
}

fn linear(x: &Matrix) -> Matrix {
    x.clone()
}

fn linear_derivative(x: &Matrix) -> Matrix {
    x.map(|_| 1.0)
}

/// A simple feedforward neural network with one hidden layer.
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Matrix,
    weights_hidden_output: Matrix,
    biases_hidden: Matrix,
    biases_output: Matrix,
    learning_rate: f64,
    activation_function: fn(&Matrix) -> Matrix,
    activation_function_derivative: fn(&Matrix) -> Matrix,
}

impl NeuralNetwork {
    /// Creates a new neural network with the given sizes for input, hidden, and output layers.
    /// The weights and biases are initialized randomly.
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden: Matrix::random(hidden_size, input_size),
            weights_hidden_output: Matrix::random(output_size, hidden_size),
            biases_hidden: Matrix::random(hidden_size, 1),
            biases_output: Matrix::random(output_size, 1),
            learning_rate: 0.01,
            activation_function: sigmoid,
            activation_function_derivative: sigmoid_derivative,
        }
    }

    /// Sets the learning rate for the neural network.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    /// Sets the activation function for the neural network.
    pub fn set_activation_function(
        &mut self,
        activation_function: fn(&Matrix) -> Matrix,
        activation_function_derivative: fn(&Matrix) -> Matrix,
    ) {
        self.activation_function = activation_function;
        self.activation_function_derivative = activation_function_derivative;
    }

    pub fn set_linear_activation(&mut self) {
        self.activation_function = linear;
        self.activation_function_derivative = linear_derivative;
    }

    pub fn set_sigmoid_activation(&mut self) {
        self.activation_function = sigmoid;
        self.activation_function_derivative = sigmoid_derivative;
    }

    pub fn set_tanh_activation(&mut self) {
        self.activation_function = tanh;
        self.activation_function_derivative = tanh_derivative;
    }

    /// Predicts the output for the given input using the neural network.
    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        // Generate the hidden outputs
        let input_matrix = Matrix::from_col_vec(input);
        let hidden_layer_input = &self.weights_input_hidden * &input_matrix + &self.biases_hidden;
        let hidden_layer_output = (self.activation_function)(&hidden_layer_input);
        // Generate the output's output
        let output_layer_input = &self.weights_hidden_output * &hidden_layer_output + &self.biases_output;
        let output_layer_output = (self.activation_function)(&output_layer_input);
        // Return the output as a vector
        output_layer_output.col(0)
    }

    /// Trains the neural network using the given input and target output.
    /// The input and target should be vectors of the same length as the input and output sizes of the network.
    /// The training process involves forward propagation and backpropagation to adjust the weights and biases.
    pub fn train(&mut self, input: Vec<f64>, target: Vec<f64>) {
        // Generate the hidden outputs
        let input = Matrix::from_col_vec(input);
        let hidden_layer_input = &self.weights_input_hidden * &input + &self.biases_hidden;
        let hidden_layer_output = (self.activation_function)(&hidden_layer_input);

        // Generate the output's outputs
        let output_layer_input = &self.weights_hidden_output * &hidden_layer_output + &self.biases_output;
        let output_layer_output = (self.activation_function)(&output_layer_input);
        
        // Create target matrix
        let target = Matrix::from_col_vec(target);

        // Calculate the error
        // ERROR = TARGET - OUTPUT
        let output_errors = target - &output_layer_output;

        // Calculate gradients
        let mut gradients = (self.activation_function_derivative)(&output_layer_output);
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
        let mut hidden_gradient = (self.activation_function_derivative)(&hidden_layer_output);
        hidden_gradient.hadamar_product(&hidden_errors);
        hidden_gradient *= self.learning_rate;

        // Calculate input -> hidden deltas
        let inputs_transposed = input.transpose();
        let weight_input_hidden_deltas = &hidden_gradient * &inputs_transposed;

        self.weights_input_hidden += &weight_input_hidden_deltas;
        // Adjust the bias by its deltas (which is just the gradient)
        self.biases_hidden += &hidden_gradient;
    }
}

pub mod tests {
    use super::*;

    #[test]
    fn it_creates_a_neural_network() {
        let m = NeuralNetwork::new(3, 5, 2);
        assert_eq!(m.input_size, 3);
        assert_eq!(m.hidden_size, 5);
        assert_eq!(m.output_size, 2);
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
        let m = NeuralNetwork::new(3, 5, 2);
        let input = vec![0.5, 0.2, 0.1];
        let output = m.predict(input.clone());
        assert_eq!(output.len(), 2);
        assert_ne!(output[0], output[1]);
    }
}
