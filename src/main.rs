mod apples;

#[macroquad::main("Macroquad Scene")]
async fn main() {
    apples::run().await;
}