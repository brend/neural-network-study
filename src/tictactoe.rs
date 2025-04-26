use std::{fs::File, io::BufReader};
use rand::seq::SliceRandom;

use crate::neural_network::NeuralNetwork;

pub fn run() {
    // Create and train a neural network with the tic-tac-toe dataset
    let nn = crate::tictactoe::train();

    // Have the user play tic-tac-toe against the neural network
    let mut board = vec![0; 9];
    loop {
        
        // Get the player's move
        let player_move_index = player_move(&board);
        // Make the move
        board[player_move_index] = 1;
        // Print the board
        print_board(&board);

        // Check for a winner
        if check_winner(&board) {
            println!("You win!");
            break;
        } else if check_draw(&board) {
            println!("It's a draw!");
            break;
        }

        // Have the neural network make its move
        let nn_move_index = nn_move(&board, &nn);
        // Make the move
        board[nn_move_index] = -1;
        // Print the board
        print_board(&board);

        // Check for a winner
        if check_winner(&board) {
            println!("NN wins!");
            break;
        } else if check_draw(&board) {
            println!("It's a draw!");
            break;
        }
    }
}

fn nn_move(board: &Vec<i32>, nn: &NeuralNetwork) -> usize {
    let inputs = board.iter().map(|i| *i as f64).collect();
    let outputs = nn.predict(inputs);
    // The output is a vector of probabilities for each cell
    // Find the index of the maximum output
    if outputs.len() != 9 {
        panic!("Invalid output from neural network.");
    }

    // Print the outputs
    for i in 0..9 {
        println!("NN position #{}: {:.2} ", i, outputs[i]);
    }

    let nn_output = outputs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    // Check if the move is valid
    if board[nn_output] != 0 {
        panic!("Invalid move by neural network.");
    }

    nn_output
}

fn player_move(board: &Vec<i32>) -> usize {
    // Get the player's move
    let mut input = String::new();
    println!("Enter your move (0-8): ");
    std::io::stdin().read_line(&mut input).unwrap();
    let move_index: usize = input.trim().parse().unwrap();

    // Check if the input is valid
    if move_index >= 9 || board[move_index] != 0 {
        println!("Invalid move, try again.");
        return player_move(board);
    }
    move_index
}

fn train() -> NeuralNetwork {
    // Read the training data from a JSON file
    let training_data = read_training_data("datasets/tic-tac-toe.json").unwrap();
    // Create a new neural network
    let mut nn = NeuralNetwork::new(9, 18, 9);

    // Set the learning rate
    nn.set_learning_rate(0.1);
    // Train the neural network
    let mut training_data = training_data.clone();
    let mut rng = rand::rng();
    let count = 100;
    for i in 0..count {
        // Shuffle the training data
        training_data.shuffle(&mut rng);
        // Train the neural network on the shuffled data
        for (input, target) in &training_data {
            nn.train(input.clone(), target.clone());
        }
        println!("Training {} % complete", i * 100 / count);
    }
    // Return the trained neural network
    nn
}

/// Reads the training data from a JSON file.
/// The JSON file should contain an array of objects, each with an "input" and "target" field,
/// both of which are arrays of numbers.
fn read_training_data(file_path: &str) -> Result<Vec<(Vec<f64>, Vec<f64>)>, Box<dyn std::error::Error>> {
    // Read the JSON file
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    // Parse the JSON data
    let data: Vec<serde_json::Value> = serde_json::from_reader(reader)?;
    // Convert the JSON data into a vector of (input, target) pairs
    let mut training_data = Vec::new();
    for item in data {
        let input: Vec<f64> = serde_json::from_value(item["input"].clone())?;
        let target: Vec<f64> = serde_json::from_value(item["target"].clone())?;
        training_data.push((input, target));
    }
    // Return the training data
    Ok(training_data)
}

fn print_board(board: &Vec<i32>) {
    for i in 0..3 {
        for j in 0..3 {
            let index = i * 3 + j;
            match board[index] {
                1 => print!("X "),
                -1 => print!("O "),
                _ => print!(". "),
            }
        }
        println!();
    }
    println!();
}

fn check_winner(board: &Vec<i32>) -> bool {
    // Check rows
    for i in 0..3 {
        if board[i * 3] == board[i * 3 + 1] && board[i * 3] == board[i * 3 + 2] && board[i * 3] != 0 {
            return true;
        }
    }
    // Check columns
    for i in 0..3 {
        if board[i] == board[i + 3] && board[i] == board[i + 6] && board[i] != 0 {
            return true;
        }
    }
    // Check diagonals
    if (board[0] == board[4] && board[0] == board[8] && board[0] != 0) ||
       (board[2] == board[4] && board[2] == board[6] && board[2] != 0) {
        return true;
    }
    false
}

fn check_draw(board: &Vec<i32>) -> bool {
    for i in 0..9 {
        if board[i] == 0 {
            return false;
        }
    }
    true
}