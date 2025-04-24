use macroquad::prelude::*;

mod matrix;

#[macroquad::main("Neural Network")]
async fn main() {    
    loop {
        clear_background(WHITE);
        next_frame().await;
    }
}