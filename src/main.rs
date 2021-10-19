#[inline]
fn sigmoid(x: f64) -> f64 {
    return 1. / (1. + ((-1. * x).exp()));
}

fn main() {
    let x = sigmoid(0.5);
    println!("{}",x.to_string());

}