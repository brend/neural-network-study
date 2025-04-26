mod matrix;
mod neural_network;
mod tictactoe;
mod simple;
mod apples;

#[macroquad::main("Macroquad Scene")]
async fn main() {
    apples::run().await;
}