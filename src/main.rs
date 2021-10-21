#[inline]
fn sigmoid(x: f64) -> f64 { 1. / (1. + ((-1. * x).exp())) }

#[inline]
fn sigmoid_prime(x: f64) -> f64 { sigmoid(x) * (1. - sigmoid(x)) }

pub struct Matrix<T> {
    data: Vec<T>,
    cols: usize,
    rows: usize,
}


impl<T: Clone> Matrix<T> {
    fn cols(&self) -> usize { self.cols }

    fn rows(&self) -> usize { self.rows }

    fn len(&self) -> usize  {self.data.len()}

    fn transpose(&self) -> Matrix<T> {
        let mut transpose: Vec<T> = Vec::new();
        for i in 0 .. self.cols{
            for j in 0 .. self.rows {
                transpose.push(self.data[j * self.cols + i].clone());
            }
        }
        Matrix::new(transpose, self.rows, self.cols)
    }

    fn slice(&self, rows: (usize, usize), cols: (usize, usize)) -> Matrix<T> {
        let mut out: Vec<T> = Vec::new();
        for i in rows.0 .. rows.1 {
            for j in cols.0 .. cols.1 {
                out.push(self.data[i * self.cols + j].clone());
            }
        }
        Matrix::new(out, rows.1 - rows.0, cols.1 - cols.0)
    }

    fn new(data: Vec<T>, rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            data,
            rows,
            cols,
        }
    }
}

impl<T> std::ops::Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        &self.data[idx[0] * self.cols + idx[1]]
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
    let mut test = vec![1, 2, 4,
                        3, 1, 2,
                        5, 3, 1];
    let matrix = Matrix::new(test, 3,3);
    let transpose: Matrix<u64> = matrix.transpose();
    //let test2 = &matrix[[0..2, 0..2]];
    let mut array: [std::ops::Range<usize>; 2] = [0..1, 0..1];
    let slice = matrix.slice((0,2), (0,1));
    for i in 0..slice.rows {
        for j in 0 .. slice.cols {
            print!(" {}", slice[[i,j]].to_string());
        }
        print!("\n");
    }
    //println!("{}", test2.to_string());
}