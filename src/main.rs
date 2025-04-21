use macroquad::prelude::*;

#[macroquad::main("Neural Network")]
async fn main() {
    let mut world = World::new(500);
    
    loop {
        clear_background(WHITE);
        world.update();
        world.draw();
        next_frame().await;
    }
}

struct Point {
    x: f32,
    y: f32,
    tag: i32,
}

impl Point {
    fn new(x: f32, y: f32, tag: i32) -> Self {
        Self { x, y, tag }
    }
}

struct World {
    points: Vec<Point>,
    divider_m: f32,
    divider_b: f32,
    perceptron: Perceptron,
    training_index: usize,
}

impl World {
    fn new(point_count: usize) -> Self {
        let mut points = Vec::new();
        let divider_m = 1.0;
        let divider_b = 0.0;
        for _ in 0..point_count {
            let x = rand::gen_range(0.0, screen_width());
            let y = rand::gen_range(0.0, screen_height());
            // points are tagged with 0 or 1 based on 
            // their position relative to the divider
            let tag = if y < divider_m * x + divider_b {
                1
            } else {
                -1
            };
            points.push(Point::new(x, y, tag));
        }
        Self {
            points,
            divider_m: 1.0,
            divider_b: 0.0,
            perceptron: Perceptron::new(2),
            training_index: 0,
        }
    }

    fn draw(&self) {
        // Draw the divider line
        let x1 = 0.0;
        let y1 = self.divider_m * x1 + self.divider_b;
        let x2 = screen_width();
        let y2 = self.divider_m * x2 + self.divider_b;
        draw_line(x1, y1, x2, y2, 2.0, BLACK);
        
        // Draw the points
        for point in &self.points {
            // Fill the points with different colors 
            // based on the perceptron prediction's accuracy
            let prediction = (self.perceptron.predict(&[point.x, point.y])).signum();
            let color = if prediction == point.tag as f32 {
                GREEN
            } else {
                RED
            };
            draw_circle(point.x, point.y, 5.0, color);

            // Draw the points' outlines with different colors 
            // based on their tag
            let color = if point.tag > 0 {
                PURPLE
            } else {
                BLUE
            };
            draw_circle_lines(point.x, point.y, 5.0, 2.0, color);
        }
    }

    fn update(&mut self) {
        // Train the perceptron
        // with the point's coordinates and tag
        if self.training_index < self.points.len() {
            let point = &self.points[self.training_index];
            self.perceptron.train(&[point.x, point.y], point.tag as f32, 0.01);
            self.training_index += 1;
        } else {
            // Reset the training index
            self.training_index = 0;
        }
    }
}

struct Perceptron {
    weights: Vec<f32>,
    bias: f32,
}

impl Perceptron {
    fn new(input_size: usize) -> Self {
        // Initialize weights and bias with random values
        // between -1.0 and 1.0
        let weights = (0..input_size).map(|_| rand::gen_range(-1.0, 1.0)).collect();
        let bias = rand::gen_range(-1.0, 1.0);
        Self { weights, bias }
    }

    fn predict(&self, inputs: &[f32]) -> f32 {
        assert!(inputs.len() == self.weights.len(), "Input size must match weights size");
        // Calculate the weighted sum of inputs
        // and add the bias
        let sum: f32 = self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum();
        sum + self.bias
    }

    fn train(&mut self, inputs: &[f32], target: f32, learning_rate: f32) {
        // Calculate the prediction
        let prediction = self.predict(inputs).signum();
        // Calculate the error
        let error = target - prediction;
        // Update weights and bias
        for (w, i) in self.weights.iter_mut().zip(inputs) {
            *w += learning_rate * error * i;
        }
        self.bias += learning_rate * error;
    }
}