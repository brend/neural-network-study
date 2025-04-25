use std::{os::unix::thread, vec};
use ::rand::{rng, seq::SliceRandom};

use macroquad::prelude::*;

mod matrix;
mod neural_network;

// #[macroquad::main("Neural Network")]
// async fn main() {
//     loop {
//         clear_background(WHITE);
//         next_frame().await;
//     }
// }

fn main() {
    let input_size = 2;
    let hidden_size = 2;
    let output_size = 2;

    // Create a new neural network
    let mut nn = neural_network::NeuralNetwork::new(input_size, hidden_size, output_size);

    nn.set_learning_rate(0.1);

    // Example dataset (XOR problem)
    let dataset = vec![
        // Meaning of the dataset:
        // First two values are the inputs (x1, x2);
        // values are 1 (True) or 0 (False).
        // Last two values are the target outputs (y1, y2);
        // y1 means True and y2 means False.
        (vec![0.0, 0.0], vec![0.0, 1.0]),
        (vec![0.0, 1.0], vec![1.0, 0.0]),
        (vec![1.0, 0.0], vec![1.0, 0.0]),
        (vec![1.0, 1.0], vec![0.0, 1.0]),
    ];

    // Train the neural network
    let mut r = rng();
    for _ in 0..10000 {
        let mut shuffled_dataset = dataset.clone();
        shuffled_dataset.shuffle(&mut r);
        for (input, target) in &shuffled_dataset {
            nn.train(input.clone(), target.clone());
        }
    }

    // Test the neural network
    for (input, target) in &dataset {
        let output = nn.predict(input.clone());
        println!("Input: {:?}, Target: {:?}, Output: {:?} ({})", input, target, output, if output[0] > output[1] { "True" } else { "False" });
    }
}