use crate::matrix::Matrix;

struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Matrix,
    weights_hidden_output: Matrix,
    biases_hidden: Matrix,
    biases_output: Matrix,
}

impl NeuralNetwork {
    /// Creates a new neural network with the given sizes for input, hidden, and output layers.
    /// The weights and biases are initialized randomly.
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden: Matrix::random(hidden_size, input_size),
            weights_hidden_output: Matrix::random(output_size, hidden_size),
            biases_hidden: Matrix::random(hidden_size, 1),
            biases_output: Matrix::random(output_size, 1),
        }
    }

    /// Trains the neural network using the provided data and labels.
    fn train(&self, data: &Vec<f64>, labels: &Vec<f64>) {
        // Training logic here
    }

    /// Predicts the output for the given input using the neural network.
    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        let input_matrix = Matrix::from_col_vec(input);
        let hidden_layer_input = &self.weights_input_hidden * &input_matrix + &self.biases_hidden;
        let hidden_layer_output = hidden_layer_input.sigmoid();
        let output_layer_input = &self.weights_hidden_output * &hidden_layer_output + &self.biases_output;
        let output_layer_output = output_layer_input.sigmoid();
        output_layer_output.col(0)
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
