use rand::Rng;
use crate::neural_network::NeuralNetwork;

pub fn run_addition() {
    // create a neural network that will be trained to add two digits
    let mut nn = NeuralNetwork::new(2, 3, 1);
    nn.set_learning_rate(0.1);
    // nn.set_linear_activation();
    // train the neural network with random data
    let mut rng = rand::rng();
    let iterations = 100000;
    for _ in 0..iterations {
        let a = rng.random_range(0..10);
        let b = rng.random_range(0..10);
        let input = vec![a as f64, b as f64];
        let target = vec![(a + b) as f64];
        nn.train(input, target);
    }
    // test the neural network with random data
    for _ in 0..10 {
        let a = rng.random_range(0..10);
        let b = rng.random_range(0..10);
        let input = vec![a as f64, b as f64];
        let target = vec![(a + b) as f64];
        let output = nn.predict(input.clone());
        println!("Input: {:?}, Target: {:?}, Output: {:?}", input, target, output);
    }
}

pub fn run_constant() {
    // create a neural network that will be trained to output a constant value
    let mut nn = NeuralNetwork::new(1, 3, 1);
    nn.set_learning_rate(0.001);
    nn.set_linear_activation(); 
    println!("{:?}", nn);
    // train the neural network with random data
    let mut rng = rand::rng();
    let iterations = 100000;
    for _ in 0..iterations {
        let input = vec![rng.random_range(0..10) as f64];
        let target = vec![5.0];
        nn.train(input, target);
    }
    println!("{:?}", nn);
    // test the neural network with random data
    for _ in 0..10 {
        let input = vec![rng.random_range(0..10) as f64];
        let target = vec![5.0];
        let output = nn.predict(input.clone());
        println!("Input: {:?}, Target: {:?}, Output: {:?}", input, target, output);
    }
}

pub fn run_identity() {
    // create a neural network that will be trained to output the same value as the input
    let mut nn = NeuralNetwork::new(1, 1, 1);
    nn.set_learning_rate(0.01);
    nn.set_linear_activation(); 
    // train the neural network with random data
    let mut rng = rand::rng();
    let iterations = 100000;
    for _ in 0..iterations {
        let input = vec![rng.random_range(0..10) as f64];
        let target = vec![input[0]];
        nn.train(input, target);
    }
    // test the neural network with random data
    for _ in 0..10 {
        let input = vec![rng.random_range(0..10) as f64];
        let target = vec![input[0]];
        let output = nn.predict(input.clone());
        println!("Input: {:?}, Target: {:?}, Output: {:?}", input, target, output);
    }
}

pub fn run_xor() {
    // create a neural network that will be trained to output the XOR of two digits
    let mut nn = NeuralNetwork::new(2, 3, 1);
    nn.set_learning_rate(0.1);
    nn.set_tanh_activation();
    // train the neural network with random data
    let iterations = 5000;
    for _ in 0..iterations {
        let a = rand::random::<bool>();
        let b = rand::random::<bool>();
        let input = vec![a as i32 as f64, b as i32 as f64];
        let target = vec![(a ^ b) as i32 as f64];
        nn.train(input, target);
    }
    // test the neural network with random data
    for _ in 0..10 {
        let a = rand::random::<bool>();
        let b = rand::random::<bool>();
        let input = vec![a as i32 as f64, b as i32 as f64];
        let target = vec![(a ^ b) as i32 as f64];
        let output = nn.predict(input.clone());
        println!("Input: {:?}, Target: {:?}, Output: {:?}", input, target, output);
    }
}