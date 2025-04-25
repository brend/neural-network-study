use macroquad::prelude::*;

mod matrix;
mod neural_network;

#[macroquad::main("Neural Network")]
async fn main() {
    loop {
        clear_background(WHITE);
        next_frame().await;
    }
}
