use macroquad::prelude::*;

#[macroquad::main("Neural Network")]
async fn main() {
    let mut world = World::new(100);

    loop {
        clear_background(BEIGE);
        world.update();
        world.draw();
        next_frame().await;
    }
}

struct World {
    points: Vec<Vec2>,
    divider_m: f32,
    divider_b: f32,
}

impl World {
    fn new(point_count: usize) -> Self {
        let mut points = Vec::new();
        for _ in 0..point_count {
            let x = rand::gen_range(0.0, screen_width());
            let y = rand::gen_range(0.0, screen_height());
            points.push(Vec2::new(x, y));
        }
        Self {
            points,
            divider_m: 1.0,
            divider_b: 0.0,
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
            let color = if point.y < self.divider_m * point.x + self.divider_b {
                PURPLE
            } else {
                BLUE
            };
            draw_circle_lines(point.x, point.y, 5.0, 2.0, color);
        }
    }

    fn update(&mut self) {
    }
}