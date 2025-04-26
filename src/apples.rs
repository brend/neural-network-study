use macroquad::prelude::*;
use crate::neural_network::NeuralNetwork;

const BASKET_WIDTH: f32 = 100.0;
const BASKET_HEIGHT: f32 = 20.0;
const APPLE_RADIUS: f32 = 10.0;
const APPLE_SPEED: f32 = 5.0;

pub async fn run() {
    let mut basket_x = screen_width() / 2.0;
    let basket_y = screen_height() - 50.0;
    let mut apple_x = rand::gen_range(0.0, screen_width() - APPLE_RADIUS * 2.0);
    let mut apple_y = 0.0;
    let mut score = 0;
    let mut apple_count = 0;

    // Initialize the neural network
    let nn = train_a_neural_network_to_catch_apples_in_a_basket();

    loop {
        clear_background(WHITE);

        // Draw basket
        draw_rectangle(basket_x, basket_y, BASKET_WIDTH, BASKET_HEIGHT, DARKGRAY);

        // Draw apple
        draw_circle(apple_x + APPLE_RADIUS, apple_y + APPLE_RADIUS, APPLE_RADIUS, RED);

        // Update apple position
        apple_y += APPLE_SPEED;

        // Check for collision with basket
        if apple_y + APPLE_RADIUS * 2.0 >= basket_y
            && apple_x + APPLE_RADIUS * 2.0 >= basket_x
            && apple_x <= basket_x + BASKET_WIDTH
        {
            score += 1;
            apple_x = rand::gen_range(0.0, screen_width() - APPLE_RADIUS * 2.0);
            apple_y = 0.0;
            apple_count += 1;
        }

        // Reset apple if it falls off screen
        if apple_y > screen_height() {
            apple_x = rand::gen_range(0.0, screen_width() - APPLE_RADIUS * 2.0);
            apple_y = 0.0;
            apple_count += 1;
        }

        // // Move basket with arrow keys
        // if is_key_down(KeyCode::Left) {
        //     basket_x -= 10.0;
        // }
        // if is_key_down(KeyCode::Right) {
        //     basket_x += 10.0;
        // }

        // Allow the neural network to control the basket
        let apple_pos = (apple_x + APPLE_RADIUS) / screen_width();
        let basket_pos = (basket_x + BASKET_WIDTH / 2.0) / screen_width();
        let input = vec![apple_pos as f64, basket_pos as f64];
        let output = nn.predict(input);
        let move_left = output[0] > 0.5;
        let move_right = output[1] > 0.5;
        if move_left {
            basket_x -= 10.0;
        }
        if move_right {
            basket_x += 10.0;
        }

        // Clamp basket position to screen bounds
        basket_x = basket_x.clamp(0.0, screen_width() - BASKET_WIDTH);

        // Display score
        draw_text(
            &format!("Score: {}/{}", score, apple_count),
            10.0,
            20.0,
            30.0,
            BLACK,
        );

        next_frame().await;
    }
}

fn train_a_neural_network_to_catch_apples_in_a_basket() -> NeuralNetwork {
    // Create a neural network with 2 inputs (apple position and basket position), 3 hidden neurons, and 1 output (basket position)
    let mut nn = NeuralNetwork::new(2, 3, 2);
    nn.set_learning_rate(0.1);

    // Train the neural network with random data
    let iterations = 10000;
    for _ in 0..iterations {
        let apple_x = rand::gen_range(0.0, screen_width());
        let basket_x = rand::gen_range(0.0, screen_width());
        // Inputs: apple position and basket position
        let input = vec![apple_x as f64 / screen_width() as f64, basket_x as f64 / screen_width() as f64];
        // Target: vector of [move_left, move_right]
        let dx = apple_x - basket_x;        
        let target = vec![-dx.signum() as f64, dx.signum() as f64];
        nn.train(input, target);
    }

    nn
}