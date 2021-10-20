#[inline]
fn sigmoid(x: f64) -> f64 { 1. / (1. + ((-1. * x).exp())) }

#[inline]
fn sigmoid_prime(x: f64) -> f64 { sigmoid(x) * (1. - sigmoid(x)) }

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
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
        product.push(dot_product(&a[start..end], &b));
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

fn loss_function(y: &[f64], sigma: &[f64]) -> f64 {
    let mut sum=0.0;
    for i in 0..sigma.len() {
        sum += (y[i] - sigma[i]).powf(2.0);
    }
    sum / sigma.len() as f64
}

fn main() {
    let weights = vec![0.5, 0.5, 0.5,0.3, 0.3, 0.3];
    let inputs = vec![0.5, 0.5, 0.5];
    let output = evaluate_layer(&weights, &inputs);
    for i in 0..output.len() {
        println!("{}", &output[i].to_string());
    }

}