use halo2curves::ff::Field;

use crate::multicore;

pub(super) fn bitreverse(mut n: usize, l: usize) -> usize {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

/// This simple utility function will parallelize an operation that is to be
/// performed over a mutable slice.
pub fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
    let n = v.len();
    let num_threads = multicore::current_num_threads();
    let mut chunk = (n as usize) / num_threads;
    if chunk < num_threads {
        chunk = 1;
    }

    multicore::scope(|scope| {
        for (chunk_num, v) in v.chunks_mut(chunk).enumerate() {
            let f = f.clone();
            scope.spawn(move |_| {
                let start = chunk_num * chunk;
                f(v, start);
            });
        }
    });
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
