

pub struct Matrix<T> {
    data: Vec<T>,
    cols: usize,
    rows: usize,
}

impl<T: Clone + std::ops::Mul> Matrix<T> {
    pub fn cols(&self) -> usize { self.cols }

    pub fn rows(&self) -> usize { self.rows }

    pub fn len(&self) -> usize  {self.data.len()}

    pub fn transpose(&self) -> Matrix<T> {
        let mut transpose: Vec<T> = Vec::new();
        for i in 0 .. self.cols{
            for j in 0 .. self.rows {
                transpose.push(self.data[j * self.cols + i].clone());
            }
        }
        Matrix::new(transpose, self.rows, self.cols)
    }

    pub fn slice(&self, rows: (usize, usize), cols: (usize, usize)) -> Matrix<T> {
        let mut out: Vec<T> = Vec::new();
        for i in rows.0 .. rows.1 {
            for j in cols.0 .. cols.1 {
                out.push(self.data[i * self.cols + j].clone());
            }
        }
        Matrix::new(out, rows.1 - rows.0, cols.1 - cols.0)
    }

    pub fn new(data: Vec<T>, rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            data,
            rows,
            cols,
        }
    }

    // fn hadamard_product(a: &[T], b: &[T]) -> Vec<T> {
    //     let mut out: Vec<T> = Vec::new();
    //     for i in 0 .. a.len() {
    //         out.push(a[i] * b[i]);
    //     }
    //     out
    // }
}

impl<T: Copy + std::ops::Mul<Output = T>> std::ops::Mul<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Matrix<T> {
        let mut out: Vec<T> = Vec::new();
        for i in 0..self.data.len() {
            out.push(rhs * self.data[i]);
        }
        Matrix {
            data: out,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T> std::ops::Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        &self.data[idx[0] * self.cols + idx[1]]
    }
}