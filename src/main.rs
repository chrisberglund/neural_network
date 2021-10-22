pub mod matrix;

pub use crate::matrix::Matrix;

#[inline]
fn sigmoid(x: f64) -> f64 { 1. / (1. + ((-1. * x).exp())) }

#[inline]
fn sigmoid_prime(x: f64) -> f64 { sigmoid(x) * (1. - sigmoid(x)) }

pub struct Scalar<T: Clone + std::ops::Mul>(T);

impl<T: Clone + std::ops::Mul> std::ops::Deref for Scalar<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut product :f64 = 0.0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}

//Multiplies two matrices. b must be a 1D array?
fn multiply_matrix(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut product: Vec<f64> = Vec::new();
    for i in 0..a.len()/b.len() {
        let start = i * b.len();
        let end = start + b.len();
        product.push(dot(&a[start..end], &b));
    }
    product
}

fn evaluate_layer(weights: &[f64], inputs: &[f64]) -> Vec<f64> {
    let mut output: Vec<f64> = Vec::new();
    let product = multiply_matrix(weights, inputs);
    for i in 0..product.len() {
        output.push(sigmoid(product[i]));
    }
    output
}

// fn layer_error(weights: Matrix<f64>, next_error: &[f64], inputs: &[f64]) -> Vec<f64> {
//     transposed_weights = weights.transpose();
//     transposed_weights *
// }

fn output_error(y:f64, a:&[f64], z: f64) -> Vec<f64> {
    let mut error: Vec<f64> = Vec::new();
    for i in 0..a.len() {
        error.push((&a[i] - y) * sigmoid_prime(z));
    }
    error
}

fn loss_function(y: &[f64], sigma: &[f64]) -> f64 {
    let mut sum=0.0;
    for i in 0..sigma.len() {
        sum += (y[i] - sigma[i]).powf(2.0);
    }
    sum / sigma.len() as f64
}

fn main() {
    let mut test = vec![1, 2, 3,
                        4, 5, 6];
    let matrix = Matrix::new(test, 2,3);
    let mut test2 = vec![7, 8,
                         9, 10,
                         11, 12];
    let matrix2 = Matrix::new(test2, 3,2);

    let mulmatrix = matrix * matrix2;
    for i in 0..mulmatrix.rows() {
        for j in 0..mulmatrix.cols() {
            print!(" {}", mulmatrix[[i,j]].to_string());
        }
        print!("\n");
    }
    //println!("{}", test2.to_string());
}