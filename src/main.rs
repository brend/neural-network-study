use neural_network_study::Matrix;
use rand::{Rng, distr::Uniform};

fn main() {
    let rows_left = 256;
    let cols_left = 128;
    let rows_right = 128;
    let cols_right = 512;
    let multiplications = 1000;
    let dist = Uniform::new(-1.0, 1.0).unwrap();

    for i in 0..multiplications {
        let rng = rand::rng();
        let data: Vec<f64> = rng.sample_iter(&dist).take(rows_left * cols_left).collect();
        let left = Matrix::from_vec(rows_left, cols_left, data);
        let rng = rand::rng();
        let data: Vec<f64> = rng
            .sample_iter(&dist)
            .take(rows_right * cols_right)
            .collect();
        let right = Matrix::from_vec(rows_right, cols_right, data);
        let _result = &left * &right;

        // if i % 100 == 0 {
        println!("iteration {}", i);
        // }
    }
}
