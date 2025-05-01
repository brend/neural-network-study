use macroquad::prelude::*;
use neural_network_study::NeuralNetwork;

const BASKET_WIDTH: f32 = 100.0;
const BASKET_HEIGHT: f32 = 20.0;
const APPLE_RADIUS: f32 = 10.0;
const APPLE_SPEED: f32 = 5.0;

#[derive(PartialEq)]
enum FruitType {
    Apple,
    Banana,
}

impl FruitType {
    fn value(&self) -> f64 {
        match self {
            FruitType::Apple => 1.0,
            FruitType::Banana => -1.0,
        }
    }
}

pub async fn run() {
    let mut basket_x = screen_width() / 2.0;
    let basket_y = screen_height() - 50.0;
    let mut fruit_x = rand::gen_range(0.0, screen_width() - APPLE_RADIUS * 2.0);
    let mut fruit_y = 0.0;
    let mut fruit_type = if rand::gen_range(0.0, 1.0) < 0.8 {
        FruitType::Apple
    } else {
        FruitType::Banana
    };
    let mut score = 0;
    let mut apple_count = 0;

    // Initialize the neural network
    let nn = train_a_neural_network_to_catch_apples_in_a_basket();

    loop {
        clear_background(WHITE);

        // Draw basket
        draw_rectangle(basket_x, basket_y, BASKET_WIDTH, BASKET_HEIGHT, DARKGRAY);

        // Draw fruit
        let fruit_color = match fruit_type {
            FruitType::Apple => RED,
            FruitType::Banana => YELLOW,
        };
        draw_circle(fruit_x + APPLE_RADIUS, fruit_y + APPLE_RADIUS - 2.0, APPLE_RADIUS, DARKGRAY);
        draw_circle(fruit_x + APPLE_RADIUS, fruit_y + APPLE_RADIUS, APPLE_RADIUS, fruit_color);

        // Update fruit position
        fruit_y += APPLE_SPEED;

        // Check for collision with basket
        if fruit_y + APPLE_RADIUS * 2.0 >= basket_y
            && fruit_x + APPLE_RADIUS * 2.0 >= basket_x
            && fruit_x <= basket_x + BASKET_WIDTH
        {
            match fruit_type {
                FruitType::Apple => score += 1,
                FruitType::Banana => score -= 1,
            }
            fruit_x = rand::gen_range(0.0, screen_width() - APPLE_RADIUS * 2.0);
            fruit_y = 0.0;
            fruit_type = if rand::gen_range(0.0, 1.0) < 0.8 {
                FruitType::Apple
            } else {
                FruitType::Banana
            };
            apple_count += 1;
        }

        // Reset fruit if it falls off screen
        if fruit_y > screen_height() {
            fruit_x = rand::gen_range(0.0, screen_width() - APPLE_RADIUS * 2.0);
            fruit_y = 0.0;
            fruit_type = if rand::gen_range(0.0, 1.0) < 0.8 {
                FruitType::Apple
            } else {
                FruitType::Banana
            };
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
        let fruit_pos = (fruit_x + APPLE_RADIUS) / screen_width();
        let basket_pos = (basket_x + BASKET_WIDTH / 2.0) / screen_width();
        let input = vec![fruit_pos as f64, fruit_type.value(), basket_pos as f64];
        let output = nn.predict(input);
        let move_left = output[0] > 0.5;
        let move_right = output[1] > 0.5;
        if move_left {
            basket_x -= 10.0;
        }
        if move_right {
            basket_x += 10.0;
        }
        // let move_left = output[0] > output[1];
        // if move_left {
        //     basket_x -= 10.0;
        // } else {
        //     basket_x += 10.0;
        // }
        
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
    // Create a neural network with 3 inputs (fruit position, fruit type, and basket position),
    // 4 hidden neurons, and 2 outputs (move left and move right)
    let mut nn = NeuralNetwork::new(3, 4, 2);
    nn.set_learning_rate(0.1);

    // Train the neural network with random data
    let iterations = 100000;
    for _ in 0..iterations {
        let fruit_x = rand::gen_range(0.0, screen_width());
        let basket_x = rand::gen_range(0.0, screen_width());
        // Inputs: fruit position, fruit type, and basket position
        let (_, fruit_type_value) = if rand::gen_range(0.0, 1.0) < 0.5 {
            (FruitType::Apple, 1.0)
        } else {
            (FruitType::Banana, -1.0)
        };
        let input = vec![fruit_x as f64 / screen_width() as f64, fruit_type_value, basket_x as f64 / screen_width() as f64];
        // Target: vector of [move_left, move_right]
        let dx = fruit_x - basket_x;        
        let target = if fruit_type_value > 0.0 {
            // It's an apple: move toward it
            vec![-dx.signum() as f64, dx.signum() as f64]
        } else {
            // It's a banana: move away from it
            vec![dx.signum() as f64, -dx.signum() as f64]
        };
        nn.train(input, target);
    }

    nn
}