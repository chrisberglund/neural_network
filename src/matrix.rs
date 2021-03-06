#[derive(Debug, PartialEq, PartialOrd, Eq)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    cols: usize,
    rows: usize,
}

impl<T: Copy + Clone + std::ops::Mul<Output=T>> Matrix<T> {
    pub fn cols(&self) -> usize { self.cols }

    pub fn rows(&self) -> usize { self.rows }

    pub fn len(&self) -> usize { self.data.len() }

    pub fn transpose(&self) -> Matrix<T> {
        let mut transpose: Vec<T> = Vec::new();
        for i in 0..self.cols {
            for j in 0..self.rows {
                transpose.push(self.data[j * self.cols + i].clone());
            }
        }
        Matrix::new(transpose, self.rows, self.cols)
    }

    pub fn slice(&self, rows: (usize, usize), cols: (usize, usize)) -> Matrix<T> {
        let mut out: Vec<T> = Vec::new();
        for i in rows.0..rows.1 {
            for j in cols.0..cols.1 {
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

    fn hadamard_product(a: &[T], b: &[T]) -> Vec<T> {
        let mut out: Vec<T> = Vec::new();
        for i in 0..a.len() {
            out.push(a[i] * b[i]);
        }
        out
    }
}


impl<T: Copy + std::ops::Mul<Output=T> + std::ops::Add<Output=T>> std::ops::Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        let mut out: Vec<T> = Vec::new();
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = self[[i, 0]] * rhs[[0, j]];
                for k in 1..rhs.rows {
                    sum = sum + self[[i, k]] * rhs[[k, j]];
                }
                out.push(sum);
            }
        }
        Matrix {
            data: out,
            rows: self.rows(),
            cols: rhs.cols(),
        }
    }
}

impl<T: Copy + std::ops::Mul<Output=T> + std::ops::Add<Output=T>> std::ops::Mul<T> for Matrix<T> {
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

#[cfg(test)]
mod test {
    use crate::Matrix;

    #[test]
    fn test_create_matrix() {
        let values = vec![0,1,2,3,4,5];
        let matrix: Matrix<u64> = Matrix::new(values,  3, 3);
        assert_eq!(matrix.data,
                   vec![0,1,2,3,4,5],
                   "New matrix did not contain correct data"
        );
        assert_eq!(matrix.rows(),
                   3,
                   "Matrix did not have correct number of rows, it had '{}' \
                   rows while 3 were expected",
                   matrix.rows());
        assert_eq!(matrix.cols(),
                   3,
                   "Matrix did not have correct number of columns, it had '{}' \
                   rows while 3 were expected",
                   matrix.cols());
    }

    #[test]
    fn test_scalar_multiplication() {
        let values = vec![0,1,2,3,4,5];
        let matrix: Matrix<u64> = Matrix::new(values,  3, 3);
        let expected_values = vec![0,3,6,9,12,15];
        let expected: Matrix<u64> = Matrix::new(expected_values,  3, 3);
        let product = matrix * 3;
        assert_eq!(product,
                   expected,
        );
    }
}