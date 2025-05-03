use neural_network_study::NeuralNetwork;

fn main() {
    let mut nn = NeuralNetwork::new(6, 5, 4);

    for i in 0..1000000 {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let target = vec![0.7, 0.8, 0.9, 0.10];
        nn.train(input, target);

        if i % 100 == 0 {
            println!("step {}", i);
        }
    }
}