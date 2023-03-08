use super::multicore;
use super::util::{bitreverse, parallelize};
use crate::arithmetic::log2_floor;
pub use ff::Field;
use group::{
    ff::{BatchInvert, PrimeField},
    Curve, Group,
};

pub use halo2curves::{CurveAffine, CurveExt};

pub const SPARSE_TWIDDLE_DEGREE: u32 = 10;
/// Performs a radix-$2$ Fast-Fourier Transformation (FFT) on a vector of size
/// $n = 2^k$, when provided `log_n` = $k$ and an element of multiplicative
/// order $n$ called `omega` ($\omega$). The result is that the vector `a`, when
/// interpreted as the coefficients of a polynomial of degree $n - 1$, is
/// transformed into the evaluations of this polynomial at each of the $n$
/// distinct powers of $\omega$. This transformation is invertible by providing
/// $\omega^{-1}$ in place of $\omega$ and dividing each resulting field element
/// by $n$.
///
/// This will use multithreading if beneficial.
pub fn best_fft<F: PrimeField>(a: &mut [F], omega: F, log_n: u32) {
    let threads = multicore::current_num_threads();
    let log_split = log2_floor(threads) as usize;
    let n = a.len() as usize;
    let sub_n = n >> log_split;
    let split_m = 1 << log_split;

    if sub_n < split_m {
        serial_fft(a, omega, log_n);
    } else {
        parallel_fft(a, omega, log_n);
    }
}

fn serial_fft<F: PrimeField>(a: &mut [F], omega: F, log_n: u32) {
    let n = a.len() as u32;
    assert_eq!(n, 1 << log_n);

    for k in 0..n as usize {
        let rk = bitreverse(k, log_n as usize);
        if k < rk {
            a.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_n {
        let w_m = omega.pow_vartime(&[u64::from(n / (2 * m)), 0, 0, 0]);

        let mut k = 0;
        while k < n {
            let mut w = F::ONE;
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t.mul_assign(&w);
                a[(k + j + m) as usize] = a[(k + j) as usize];
                a[(k + j + m) as usize].sub_assign(&t);
                a[(k + j) as usize].add_assign(&t);
                w *= &w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn serial_split_fft<F: PrimeField>(
    a: &mut [F],
    twiddle_lut: &[F],
    twiddle_scale: usize,
    log_n: u32,
) {
    let n = a.len() as u32;
    assert_eq!(n, 1 << log_n);

    let mut m = 1;
    for _ in 0..log_n {
        let omega_idx = twiddle_scale * n as usize / (2 * m as usize); // 1/2, 1/4, 1/8, ...
        let low_idx = omega_idx % (1 << SPARSE_TWIDDLE_DEGREE);
        let high_idx = omega_idx >> SPARSE_TWIDDLE_DEGREE;
        let mut w_m = twiddle_lut[low_idx];
        if high_idx > 0 {
            w_m = w_m * twiddle_lut[(1 << SPARSE_TWIDDLE_DEGREE) + high_idx];
        }

        let mut k = 0;
        while k < n {
            let mut w = F::ONE;
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t.mul_assign(&w);
                a[(k + j + m) as usize] = a[(k + j) as usize];
                a[(k + j + m) as usize].sub_assign(&t);
                a[(k + j) as usize].add_assign(&t);
                w *= &w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn split_radix_fft<F: PrimeField>(
    tmp: &mut [F],
    a: &[F],
    twiddle_lut: &[F],
    n: usize,
    sub_fft_offset: usize,
    log_split: usize,
) {
    let split_m = 1 << log_split;
    let sub_n = n >> log_split;

    // we use out-place bitreverse here, split_m <= num_threads, so the buffer spase is small
    // and it's is good for data locality
    let mut t1 = vec![F::ZERO; split_m];
    // if unsafe code is allowed, a 10% performance improvement can be achieved
    // let mut t1: Vec<G> = Vec::with_capacity(split_m as usize);
    // unsafe{ t1.set_len(split_m as usize); }
    for i in 0..split_m {
        t1[bitreverse(i, log_split)] = a[(i * sub_n + sub_fft_offset)];
    }
    serial_split_fft(&mut t1, twiddle_lut, sub_n, log_split as u32);

    let sparse_degree = SPARSE_TWIDDLE_DEGREE;
    let omega_idx = sub_fft_offset as usize;
    let low_idx = omega_idx % (1 << sparse_degree);
    let high_idx = omega_idx >> sparse_degree;
    let mut omega = twiddle_lut[low_idx];
    if high_idx > 0 {
        omega = omega * twiddle_lut[(1 << sparse_degree) + high_idx];
    }
    let mut w_m = F::ONE;
    for i in 0..split_m {
        t1[i].mul_assign(&w_m);
        tmp[i] = t1[i];
        w_m = w_m * omega;
    }
}

pub fn generate_twiddle_lookup_table<F: Field>(
    omega: F,
    log_n: u32,
    sparse_degree: u32,
    with_last_level: bool,
) -> Vec<F> {
    let without_last_level = !with_last_level;
    let is_lut_len_large = sparse_degree > log_n;

    // dense
    if is_lut_len_large {
        let mut twiddle_lut = vec![F::ZERO; (1 << log_n) as usize];
        parallelize(&mut twiddle_lut, |twiddle_lut, start| {
            let mut w_n = omega.pow_vartime(&[start as u64, 0, 0, 0]);
            for twiddle_lut in twiddle_lut.iter_mut() {
                *twiddle_lut = w_n;
                w_n = w_n * omega;
            }
        });
        return twiddle_lut;
    }

    // sparse
    let low_degree_lut_len = 1 << sparse_degree;
    let high_degree_lut_len = 1 << (log_n - sparse_degree - without_last_level as u32);
    let mut twiddle_lut = vec![F::ZERO; (low_degree_lut_len + high_degree_lut_len) as usize];
    parallelize(
        &mut twiddle_lut[..low_degree_lut_len],
        |twiddle_lut, start| {
            let mut w_n = omega.pow_vartime(&[start as u64, 0, 0, 0]);
            for twiddle_lut in twiddle_lut.iter_mut() {
                *twiddle_lut = w_n;
                w_n = w_n * omega;
            }
        },
    );
    let high_degree_omega = omega.pow_vartime(&[(1 << sparse_degree) as u64, 0, 0, 0]);
    parallelize(
        &mut twiddle_lut[low_degree_lut_len..],
        |twiddle_lut, start| {
            let mut w_n = high_degree_omega.pow_vartime(&[start as u64, 0, 0, 0]);
            for twiddle_lut in twiddle_lut.iter_mut() {
                *twiddle_lut = w_n;
                w_n = w_n * high_degree_omega;
            }
        },
    );
    twiddle_lut
}

pub fn parallel_fft<F: PrimeField>(a: &mut [F], omega: F, log_n: u32) {
    let n = a.len() as usize;
    assert_eq!(n, 1 << log_n);

    let log_split = log2_floor(multicore::current_num_threads()) as usize;
    let split_m = 1 << log_split;
    let sub_n = n >> log_split as usize;
    let twiddle_lut = generate_twiddle_lookup_table(omega, log_n, SPARSE_TWIDDLE_DEGREE, true);

    // split fft
    let mut tmp = vec![F::ZERO; n];
    // if unsafe code is allowed, a 10% performance improvement can be achieved
    // let mut tmp: Vec<G> = Vec::with_capacity(n);
    // unsafe{ tmp.set_len(n); }
    multicore::scope(|scope| {
        let a = &*a;
        let twiddle_lut = &*twiddle_lut;
        for (chunk_idx, tmp) in tmp.chunks_mut(sub_n).enumerate() {
            scope.spawn(move |_| {
                let split_fft_offset = (chunk_idx * sub_n) >> log_split;
                for (i, tmp) in tmp.chunks_mut(split_m).enumerate() {
                    let split_fft_offset = split_fft_offset + i;
                    split_radix_fft(tmp, a, twiddle_lut, n, split_fft_offset, log_split);
                }
            });
        }
    });

    // shuffle
    parallelize(a, |a, start| {
        for (idx, a) in a.iter_mut().enumerate() {
            let idx = start + idx;
            let i = idx / sub_n;
            let j = idx % sub_n;
            *a = tmp[j * split_m + i];
        }
    });

    // sub fft
    let new_omega = omega.pow_vartime(&[split_m as u64, 0, 0, 0]);
    multicore::scope(|scope| {
        for a in a.chunks_mut(sub_n) {
            scope.spawn(move |_| {
                serial_fft(a, new_omega, log_n - log_split as u32);
            });
        }
    });

    // copy & unshuffle
    let mask = (1 << log_split) - 1;
    parallelize(&mut tmp, |tmp, start| {
        for (idx, tmp) in tmp.iter_mut().enumerate() {
            let idx = start + idx;
            *tmp = a[idx];
        }
    });
    parallelize(a, |a, start| {
        for (idx, a) in a.iter_mut().enumerate() {
            let idx = start + idx;
            *a = tmp[sub_n * (idx & mask) + (idx >> log_split)];
        }
    });
}
