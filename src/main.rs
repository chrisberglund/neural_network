#[inline]
fn sigmoid(x: f64) -> f64 { 1. / (1. + ((-1. * x).exp())) }

#[inline]
fn sigmoid_prime(x: f64) -> f64 { sigmoid(x) * (1. - sigmoid(x)) }

fn evaluate_layer(weights: &[f64], inputs: &[f64]) -> f64 {
    let mut product :f64 = 0.0;
    for i in 0..inputs.len() {
        product += weights[i] * inputs[i];
    }
    sigmoid(product)
}

fn main() {
    let weights = [0.5, 0.5, 0.5];
    let inputs = [0.5, 0.5, 0.5];

    let output = evaluate_layer(&weights, &inputs);
    println!("{}",output.to_string());

}