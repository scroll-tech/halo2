//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

use super::multicore;
pub use ff::Field;
use group::{
    ff::{BatchInvert, PrimeField},
    Curve, Group as _,
};

pub use halo2curves::{CurveAffine, CurveExt, FieldExt, Group};

pub const SPARSE_TWIDDLE_DEGREE: u32 = 10;

fn multiexp_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C], acc: &mut C::Curve) {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();

    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (f64::from(bases.len() as u32)).ln().ceil() as usize
    };

    fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
        let skip_bits = segment * c;
        let skip_bytes = skip_bits / 8;

        if skip_bytes >= 32 {
            return 0;
        }

        let mut v = [0; 8];
        for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
            *v = *o;
        }

        let mut tmp = u64::from_le_bytes(v);
        tmp >>= skip_bits - (skip_bytes * 8);
        tmp = tmp % (1 << c);

        tmp as usize
    }

    let segments = (256 / c) + 1;

    for current_segment in (0..segments).rev() {
        for _ in 0..c {
            *acc = acc.double();
        }

        #[derive(Clone, Copy)]
        enum Bucket<C: CurveAffine> {
            None,
            Affine(C),
            Projective(C::Curve),
        }

        impl<C: CurveAffine> Bucket<C> {
            fn add_assign(&mut self, other: &C) {
                *self = match *self {
                    Bucket::None => Bucket::Affine(*other),
                    Bucket::Affine(a) => Bucket::Projective(a + *other),
                    Bucket::Projective(mut a) => {
                        a += *other;
                        Bucket::Projective(a)
                    }
                }
            }

            fn add(self, mut other: C::Curve) -> C::Curve {
                match self {
                    Bucket::None => other,
                    Bucket::Affine(a) => {
                        other += a;
                        other
                    }
                    Bucket::Projective(a) => other + &a,
                }
            }
        }

        let mut buckets: Vec<Bucket<C>> = vec![Bucket::None; (1 << c) - 1];

        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let coeff = get_at::<C::Scalar>(current_segment, c, coeff);
            if coeff != 0 {
                buckets[coeff - 1].add_assign(base);
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = C::Curve::identity();
        for exp in buckets.into_iter().rev() {
            running_sum = exp.add(running_sum);
            *acc = *acc + &running_sum;
        }
    }
}

/// Performs a small multi-exponentiation operation.
/// Uses the double-and-add algorithm with doublings shared across points.
pub fn small_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();
    let mut acc = C::Curve::identity();

    // for byte idx
    for byte_idx in (0..32).rev() {
        // for bit idx
        for bit_idx in (0..8).rev() {
            acc = acc.double();
            // for each coeff
            for coeff_idx in 0..coeffs.len() {
                let byte = coeffs[coeff_idx].as_ref()[byte_idx];
                if ((byte >> bit_idx) & 1) != 0 {
                    acc += bases[coeff_idx];
                }
            }
        }
    }

    acc
}

/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
pub fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    assert_eq!(coeffs.len(), bases.len());

    let num_threads = multicore::current_num_threads();
    if coeffs.len() > num_threads {
        let chunk = coeffs.len() / num_threads;
        let num_chunks = coeffs.chunks(chunk).len();
        let mut results = vec![C::Curve::identity(); num_chunks];
        multicore::scope(|scope| {
            let chunk = coeffs.len() / num_threads;

            for ((coeffs, bases), acc) in coeffs
                .chunks(chunk)
                .zip(bases.chunks(chunk))
                .zip(results.iter_mut())
            {
                scope.spawn(move |_| {
                    multiexp_serial(coeffs, bases, acc);
                });
            }
        });
        results.iter().fold(C::Curve::identity(), |a, b| a + b)
    } else {
        let mut acc = C::Curve::identity();
        multiexp_serial(coeffs, bases, &mut acc);
        acc
    }
}

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
pub fn best_fft<G: Group>(a: &mut [G], omega: G::Scalar, log_n: u32) {
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

fn bitreverse(mut n: usize, l: usize) -> usize {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

fn serial_fft<G: Group>(a: &mut [G], omega: G::Scalar, log_n: u32) {
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
            let mut w = G::Scalar::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t.group_scale(&w);
                a[(k + j + m) as usize] = a[(k + j) as usize];
                a[(k + j + m) as usize].group_sub(&t);
                a[(k + j) as usize].group_add(&t);
                w *= &w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn serial_split_fft<G: Group>(a: &mut [G], twiddle_lut: &[G::Scalar], twiddle_scale: usize, log_n: u32) {
    let n = a.len() as u32;
    assert_eq!(n, 1 << log_n);

    let mut m = 1;
    for _ in 0..log_n {
        let omega_idx = twiddle_scale * n as usize / (2 * m as usize); // 1/2, 1/4, 1/8, ...
        let low_idx = omega_idx%(1<<SPARSE_TWIDDLE_DEGREE);
        let high_idx = omega_idx>>SPARSE_TWIDDLE_DEGREE;
        let mut w_m = twiddle_lut[low_idx];
        if high_idx > 0 {
            w_m = w_m*twiddle_lut[(1<<SPARSE_TWIDDLE_DEGREE) + high_idx];
        }

        let mut k = 0;
        while k < n {
            let mut w = G::Scalar::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t.group_scale(&w);
                a[(k + j + m) as usize] = a[(k + j) as usize];
                a[(k + j + m) as usize].group_sub(&t);
                a[(k + j) as usize].group_add(&t);
                w *= &w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn split_radix_fft<G: Group>(tmp: &mut [G], a: &[G], twiddle_lut: &[G::Scalar], n: usize, sub_fft_offset: usize, log_split: usize) {
    let split_m = 1 << log_split;
    let sub_n = n >> log_split;

    // we use out-place bitreverse here, split_m <= num_threads, so the buffer spase is small
    // and it's is good for data locality
    let mut t1 = vec![G::group_zero(); split_m];
    // if unsafe code is allowed, a 10% performance improvement can be achieved
    // let mut t1: Vec<G> = Vec::with_capacity(split_m as usize);
    // unsafe{ t1.set_len(split_m as usize); }
    for i in 0..split_m {
        t1[bitreverse(i,log_split)] = a[(i*sub_n + sub_fft_offset)];
    }
    serial_split_fft(&mut t1, twiddle_lut, sub_n, log_split as u32);

    let sparse_degree = SPARSE_TWIDDLE_DEGREE;
    let omega_idx = sub_fft_offset as usize;
    let low_idx = omega_idx%(1<<sparse_degree);
    let high_idx = omega_idx>>sparse_degree;
    let mut omega = twiddle_lut[low_idx];
    if high_idx > 0 {
        omega = omega*twiddle_lut[(1<<sparse_degree) + high_idx];
    }
    let mut w_m = G::Scalar::one();
    for i in 0..split_m {
        t1[i].group_scale(&w_m);
        tmp[i] = t1[i];
        w_m = w_m*omega;
    }
}

pub fn generate_twiddle_lookup_table<F: Field>(omega: F, log_n: u32, sparse_degree: u32, with_last_level: bool) -> Vec<F> {
    let without_last_level = !with_last_level;
    let is_lut_len_large = sparse_degree > log_n;

    // dense
    if is_lut_len_large {
        let mut twiddle_lut = vec![F::zero(); (1<<log_n) as usize];
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
    let low_degree_lut_len = 1<<sparse_degree;
    let high_degree_lut_len = 1 << (log_n - sparse_degree - without_last_level as u32);
    let mut twiddle_lut = vec![F::zero(); (low_degree_lut_len + high_degree_lut_len) as usize];
    parallelize(&mut twiddle_lut[..low_degree_lut_len], |twiddle_lut, start| {
        let mut w_n = omega.pow_vartime(&[start as u64, 0, 0, 0]);
        for twiddle_lut in twiddle_lut.iter_mut() {
            *twiddle_lut = w_n;
            w_n = w_n * omega;
        }
    });
    let high_degree_omega = omega.pow_vartime(&[(1<<sparse_degree) as u64, 0, 0, 0]);
    parallelize(&mut twiddle_lut[low_degree_lut_len..], |twiddle_lut, start| {
        let mut w_n = high_degree_omega.pow_vartime(&[start as u64, 0, 0, 0]);
        for twiddle_lut in twiddle_lut.iter_mut() {
            *twiddle_lut = w_n;
            w_n = w_n * high_degree_omega;
        }
    });
    twiddle_lut
}

pub fn parallel_fft<G: Group>(a: &mut [G], omega: G::Scalar, log_n: u32) {
    let n = a.len() as usize;
    assert_eq!(n, 1 << log_n);

    let log_split = log2_floor(multicore::current_num_threads()) as usize;
    let split_m = 1 << log_split;
    let sub_n = n >> log_split as usize;
    let twiddle_lut = generate_twiddle_lookup_table(omega, log_n, SPARSE_TWIDDLE_DEGREE, true);

    // split fft
    let mut tmp = vec![G::group_zero(); n];
    // if unsafe code is allowed, a 10% performance improvement can be achieved
    // let mut tmp: Vec<G> = Vec::with_capacity(n);
    // unsafe{ tmp.set_len(n); }
    multicore::scope(|scope| {
        let a = &*a;
        let twiddle_lut = &*twiddle_lut;
        for (chunk_idx, tmp) in tmp.chunks_mut(sub_n).enumerate() {
            scope.spawn(move |_| {
                let split_fft_offset = chunk_idx * sub_n >> log_split;
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
            *a = tmp[j*split_m + i];
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
            *a = tmp[sub_n*(idx & mask) + (idx >> log_split)];
        }
    });
}

/// Convert coefficient bases group elements to lagrange basis by inverse FFT.
pub fn g_to_lagrange<C: CurveAffine>(g_projective: Vec<C::Curve>, k: u32) -> Vec<C> {
    let n_inv = C::Scalar::TWO_INV.pow_vartime(&[k as u64, 0, 0, 0]);
    let mut omega_inv = C::Scalar::ROOT_OF_UNITY_INV;
    for _ in k..C::Scalar::S {
        omega_inv = omega_inv.square();
    }

    let mut g_lagrange_projective = g_projective;
    best_fft(&mut g_lagrange_projective, omega_inv, k);
    parallelize(&mut g_lagrange_projective, |g, _| {
        for g in g.iter_mut() {
            *g *= n_inv;
        }
    });

    let mut g_lagrange = vec![C::identity(); 1 << k];
    parallelize(&mut g_lagrange, |g_lagrange, starts| {
        C::Curve::batch_normalize(
            &g_lagrange_projective[starts..(starts + g_lagrange.len())],
            g_lagrange,
        );
    });

    g_lagrange
}

/// This evaluates a provided polynomial (in coefficient form) at `point`.
pub fn eval_polynomial<F: Field>(poly: &[F], point: F) -> F {
    fn evaluate<F: Field>(poly: &[F], point: F) -> F {
        poly.iter()
            .rev()
            .fold(F::zero(), |acc, coeff| acc * point + coeff)
    }
    let n = poly.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(poly, point)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::zero(); num_threads];
        multicore::scope(|scope| {
            for (chunk_idx, (out, poly)) in
                parts.chunks_mut(1).zip(poly.chunks(chunk_size)).enumerate()
            {
                scope.spawn(move |_| {
                    let start = chunk_idx * chunk_size;
                    out[0] = evaluate(poly, point) * point.pow_vartime(&[start as u64, 0, 0, 0]);
                });
            }
        });
        parts.iter().fold(F::zero(), |acc, coeff| acc + coeff)
    }
}

/// This computes the inner product of two vectors `a` and `b`.
///
/// This function will panic if the two vectors are not the same size.
pub fn compute_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    // TODO: parallelize?
    assert_eq!(a.len(), b.len());

    let mut acc = F::zero();
    for (a, b) in a.iter().zip(b.iter()) {
        acc += (*a) * (*b);
    }

    acc
}

/// Divides polynomial `a` in `X` by `X - b` with
/// no remainder.
pub fn kate_division<'a, F: Field, I: IntoIterator<Item = &'a F>>(a: I, mut b: F) -> Vec<F>
where
    I::IntoIter: DoubleEndedIterator + ExactSizeIterator,
{
    b = -b;
    let a = a.into_iter();

    let mut q = vec![F::zero(); a.len() - 1];

    let mut tmp = F::zero();
    for (q, r) in q.iter_mut().rev().zip(a.rev()) {
        let mut lead_coeff = *r;
        lead_coeff.sub_assign(&tmp);
        *q = lead_coeff;
        tmp = lead_coeff;
        tmp.mul_assign(&b);
    }

    q
}

/// This simple utility function will parallelize an operation that is to be
/// performed over a mutable slice.
pub fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
    let n = v.len();
    let num_threads = multicore::current_num_threads();
    let mut chunk = (n as usize) / num_threads;
    if chunk < num_threads {
        chunk = n as usize;
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

fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn lagrange_interpolate<F: FieldExt>(points: &[F], evals: &[F]) -> Vec<F> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        // Constant polynomial
        vec![evals[0]]
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        for (j, x_j) in points.iter().enumerate() {
            let mut denom = Vec::with_capacity(points.len() - 1);
            for x_k in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
            {
                denom.push(*x_j - x_k);
            }
            denoms.push(denom);
        }
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();

        let mut final_poly = vec![F::zero(); points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::one());
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms.into_iter())
            {
                product.resize(tmp.len() + 1, F::zero());
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::zero()))
                    .zip(std::iter::once(&F::zero()).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(tmp.into_iter()) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
        final_poly
    }
}

pub(crate) fn evaluate_vanishing_polynomial<F: FieldExt>(roots: &[F], z: F) -> F {
    fn evaluate<F: FieldExt>(roots: &[F], z: F) -> F {
        roots.iter().fold(F::one(), |acc, point| (z - point) * acc)
    }
    let n = roots.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(roots, z)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::one(); num_threads];
        multicore::scope(|scope| {
            for (out, roots) in parts.chunks_mut(1).zip(roots.chunks(chunk_size)) {
                scope.spawn(move |_| out[0] = evaluate(roots, z));
            }
        });
        parts.iter().fold(F::one(), |acc, part| acc * part)
    }
}

pub(crate) fn powers<F: FieldExt>(base: F) -> impl Iterator<Item = F> {
    std::iter::successors(Some(F::one()), move |power| Some(base * power))
}

#[cfg(test)]
use rand_core::OsRng;

#[cfg(test)]
use crate::halo2curves::pasta::Fp;

#[test]
fn test_lagrange_interpolate() {
    let rng = OsRng;

    let points = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();
    let evals = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();

    for coeffs in 0..5 {
        let points = &points[0..coeffs];
        let evals = &evals[0..coeffs];

        let poly = lagrange_interpolate(points, evals);
        assert_eq!(poly.len(), points.len());

        for (point, eval) in points.iter().zip(evals) {
            assert_eq!(eval_polynomial(&poly, *point), *eval);
        }
    }
}
