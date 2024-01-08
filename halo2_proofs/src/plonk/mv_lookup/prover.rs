use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeTheta, ChallengeX, Error, ProvingKey,
};
use super::Argument;
use crate::plonk::evaluation::evaluate;
use crate::{
    arithmetic::{eval_polynomial, parallelize, CurveAffine},
    poly::{
        commitment::{Blind, Params},
        Coeff, EvaluationDomain, LagrangeCoeff, Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ark_std::{end_timer, start_timer};
use ff::{PrimeField, WithSmallOrderMulGroup};
use group::{ff::Field, Curve};
use rand_core::RngCore;
use std::{
    iter,
    ops::{Mul, MulAssign},
};

use crate::arithmetic::{par_invert, parallelize_internal};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};

#[derive(Debug)]
pub(in crate::plonk) struct Prepared<C: CurveAffine> {
    compressed_inputs_expressions: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    m_values: Polynomial<C::Scalar, LagrangeCoeff>,
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine> {
    pub(in crate::plonk) m_poly: Polynomial<C::Scalar, Coeff>,
    pub(in crate::plonk) phi_poly: Polynomial<C::Scalar, Coeff>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Committed<C>,
}

impl<F: PrimeField + WithSmallOrderMulGroup<3> + Ord> Argument<F> {
    pub(in crate::plonk) fn prepare<
        'a,
        'params: 'a,
        C,
        P: Params<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        &self,
        pk: &ProvingKey<C>,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        challenges: &'a [C::Scalar],
        #[cfg(feature = "sanity-checks")] mut rng: R, // in case we want to blind (do we actually need zk?)
        #[cfg(not(feature = "sanity-checks"))] rng: R,
        transcript: &mut T,
    ) -> Result<Prepared<C>, Error>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        let prepare_time = start_timer!(|| format!(
            "prepare m(X) (inputs={:?}, table={})",
            self.inputs_expressions
                .iter()
                .map(|e| e.len())
                .collect::<Vec<usize>>(),
            self.table_expressions.len()
        ));
        // Closure to get values of expressions and compress them
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            let compressed_expression = expressions
                .iter()
                .map(|expression| {
                    pk.vk.domain.lagrange_from_vec(evaluate(
                        expression,
                        params.n() as usize,
                        1,
                        fixed_values,
                        advice_values,
                        instance_values,
                        challenges,
                    ))
                })
                .fold(domain.empty_lagrange(), |acc, expression| {
                    acc * *theta + &expression
                });
            compressed_expression
        };

        // Get values of input expressions involved in the lookup and compress them
        let compressed_inputs_expressions: Vec<_> = self
            .inputs_expressions
            .iter()
            .map(|input_expressions| compress_expressions(input_expressions))
            .collect();

        // Get values of table expressions involved in the lookup and compress them
        let compressed_table_expression = compress_expressions(&self.table_expressions);

        let blinding_factors = pk.vk.cs.blinding_factors();

        // compute m(X)
        let tivm_time = start_timer!(|| "table index value mapping");
        let mut sorted_table_with_indices = compressed_table_expression
            .iter()
            .take(params.n() as usize - blinding_factors - 1)
            .enumerate()
            .map(|(i, t)| (t, i))
            .collect::<Vec<_>>();
        sorted_table_with_indices.par_sort_by_key(|(&t, _)| t);
        end_timer!(tivm_time);

        let m_time = start_timer!(|| "m(X) values");
        let m_values: Vec<F> = {
            use std::sync::atomic::{AtomicU64, Ordering};
            let m_values: Vec<AtomicU64> = (0..params.n()).map(|_| AtomicU64::new(0)).collect();

            for compressed_input_expression in compressed_inputs_expressions.iter() {
                let _ = compressed_input_expression
                    .par_iter()
                    .take(params.n() as usize - blinding_factors - 1)
                    .try_for_each(|fi| -> Result<(), Error> {
                        let index = sorted_table_with_indices
                            .binary_search_by_key(&fi, |&(t, _)| t)
                            .map_err(|_| Error::ConstraintSystemFailure)?;
                        let index = sorted_table_with_indices[index].1;

                        m_values[index].fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    });
            }

            m_values
                .par_iter()
                .map(|mi| F::from(mi.load(Ordering::Relaxed)))
                .collect()
        };
        end_timer!(m_time);
        let m_values = pk.vk.domain.lagrange_from_vec(m_values);

        #[cfg(feature = "sanity-checks")]
        {
            // check that m is zero after blinders
            let invalid_ms = m_values
                .iter()
                .skip(params.n() as usize - blinding_factors)
                .collect::<Vec<_>>();
            assert_eq!(invalid_ms.len(), blinding_factors);
            for mi in invalid_ms {
                assert_eq!(*mi, C::Scalar::ZERO);
            }

            // check sums
            let alpha = C::Scalar::random(&mut rng);
            let cs_input_sum =
                |compressed_input_expression: &Polynomial<C::Scalar, LagrangeCoeff>| {
                    let mut lhs_sum = C::Scalar::ZERO;
                    for &fi in compressed_input_expression
                        .iter()
                        .take(params.n() as usize - blinding_factors - 1)
                    {
                        lhs_sum += (fi + alpha).invert().unwrap();
                    }

                    lhs_sum
                };

            let mut lhs_sum = C::Scalar::ZERO;

            for compressed_input_expression in compressed_inputs_expressions.iter() {
                lhs_sum += cs_input_sum(compressed_input_expression);
            }

            let mut rhs_sum = C::Scalar::ZERO;
            for (&ti, &mi) in compressed_table_expression.iter().zip(m_values.iter()) {
                rhs_sum += mi * (ti + alpha).invert().unwrap();
            }

            assert_eq!(lhs_sum, rhs_sum);
        }

        // commit to m(X)
        // TODO: should we use zero instead?
        let blind = Blind(C::Scalar::random(rng));
        let m_commitment = params.commit_lagrange(&m_values, blind).to_affine();

        // write commitment of m(X) to transcript
        transcript.write_point(m_commitment)?;

        end_timer!(prepare_time);

        Ok(Prepared {
            compressed_inputs_expressions,
            compressed_table_expression,
            m_values,
        })
    }
}

impl<C: CurveAffine> Prepared<C> {
    pub(in crate::plonk) fn commit_grand_sum<
        'params,
        P: Params<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        self,
        pk: &ProvingKey<C>,
        params: &P,
        beta: ChallengeBeta<C>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        /*
            φ_i(X) = f_i(X) + beta
            τ(X) = t(X) + beta
            LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
            RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
        */
        let lookup_commit_time = start_timer!(|| "commit_grand_sum");

        // ∑ 1/(φ_i(X))
        let inputs_log_drv_time = start_timer!(|| "inputs_log_derivative");
        let mut inputs_log_derivatives = vec![C::Scalar::ZERO; params.n() as usize];
        for compressed_input_expression in self.compressed_inputs_expressions.iter() {
            let mut input_log_derivatives = vec![C::Scalar::ZERO; params.n() as usize];

            parallelize(
                &mut input_log_derivatives,
                |input_log_derivatives, start| {
                    for (input_log_derivative, fi) in input_log_derivatives
                        .iter_mut()
                        .zip(compressed_input_expression[start..].iter())
                    {
                        *input_log_derivative = *beta + fi;
                    }
                },
            );
            let inputs_inv_time = start_timer!(|| "batch invert");
            par_invert(input_log_derivatives.as_mut_slice());
            end_timer!(inputs_inv_time);

            // TODO: remove last blinders from this
            for i in 0..params.n() as usize {
                inputs_log_derivatives[i] += input_log_derivatives[i];
            }
        }
        end_timer!(inputs_log_drv_time);

        // 1 / τ(X)
        let table_log_drv_time = start_timer!(|| "table log derivative");
        let mut table_log_derivatives = vec![C::Scalar::ZERO; params.n() as usize];
        parallelize(
            &mut table_log_derivatives,
            |table_log_derivatives, start| {
                for (table_log_derivative, ti) in table_log_derivatives
                    .iter_mut()
                    .zip(self.compressed_table_expression[start..].iter())
                {
                    *table_log_derivative = *beta + ti;
                }
            },
        );

        let table_inv_time = start_timer!(|| "table batch invert");
        par_invert(table_log_derivatives.as_mut_slice());
        end_timer!(table_inv_time);
        end_timer!(table_log_drv_time);

        let log_drv_diff_time = start_timer!(|| "log derivatives diff");
        // (Σ 1/(φ_i(X)) - m(X) / τ(X))
        let mut log_derivatives_diff = vec![C::Scalar::ZERO; params.n() as usize];
        parallelize(&mut log_derivatives_diff, |log_derivatives_diff, start| {
            for (((log_derivative_diff, fi), ti), mi) in log_derivatives_diff
                .iter_mut()
                .zip(inputs_log_derivatives[start..].iter())
                .zip(table_log_derivatives[start..].iter())
                .zip(self.m_values[start..].iter())
            {
                // (Σ 1/(φ_i(X)) - m(X) / τ(X))
                *log_derivative_diff = *fi - *mi * *ti;
            }
        });
        end_timer!(log_drv_diff_time);

        // Compute the evaluations of the lookup grand sum polynomial
        // over our domain, starting with phi[0] = 0
        let blinding_factors = pk.vk.cs.blinding_factors();
        let phi_time = start_timer!(|| "par_scan(log_derivatives_diff)");
        let phi = {
            // parallelized version of log_derivatives_diff.scan()
            let active_size = params.n() as usize - blinding_factors;
            let mut grand_sum = iter::once(C::Scalar::ZERO)
                .chain(log_derivatives_diff)
                .take(active_size)
                .collect::<Vec<_>>();
            // TODO: remove the implicit assumption that parallelize() split the grand_sum
            //      into segments that each has `chunk` elements except the last.
            let segment_starts = parallelize_internal(&mut grand_sum, |segment_grand_sum, _| {
                for i in 1..segment_grand_sum.len() {
                    segment_grand_sum[i] += segment_grand_sum[i - 1];
                }
            });
            let mut segment_sum = vec![C::Scalar::ZERO; grand_sum.len()];
            for i in 1..segment_starts.len() {
                segment_sum[segment_starts[i]] =
                    segment_sum[segment_starts[i - 1]] + grand_sum[segment_starts[i] - 1];
            }
            parallelize(&mut grand_sum, |grand_sum, start| {
                let prefix_sum = segment_sum[start];
                for v in grand_sum.iter_mut() {
                    *v += prefix_sum;
                }
            });
            grand_sum
                .into_iter()
                .chain((0..blinding_factors).map(|_| C::Scalar::random(&mut rng)))
                .collect::<Vec<_>>()
        };
        end_timer!(phi_time);
        assert_eq!(phi.len(), params.n() as usize);
        let phi = pk.vk.domain.lagrange_from_vec(phi);

        #[cfg(feature = "sanity-checks")]
        // This test works only with intermediate representations in this method.
        // It can be used for debugging purposes.
        {
            // While in Lagrange basis, check that product is correctly constructed
            let u = (params.n() as usize) - (blinding_factors + 1);

            /*
                φ_i(X) = f_i(X) + α
                τ(X) = t(X) + α
                LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
            */

            // q(X) = LHS - RHS mod zH(X)
            for i in 0..u {
                // Π(φ_i(X))
                let fi_prod = || {
                    let mut prod = C::Scalar::ONE;
                    for compressed_input_expression in self.compressed_inputs_expressions.iter() {
                        prod *= *beta + compressed_input_expression[i];
                    }

                    prod
                };

                let fi_log_derivative = || {
                    let mut sum = C::Scalar::ZERO;
                    for compressed_input_expression in self.compressed_inputs_expressions.iter() {
                        sum += (*beta + compressed_input_expression[i]).invert().unwrap();
                    }

                    sum
                };

                // LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                let lhs = {
                    (*beta + self.compressed_table_expression[i])
                        * fi_prod()
                        * (phi[i + 1] - phi[i])
                };

                // RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
                let rhs = {
                    (*beta + self.compressed_table_expression[i])
                        * fi_prod()
                        * (fi_log_derivative()
                            - self.m_values[i]
                                * (*beta + self.compressed_table_expression[i])
                                    .invert()
                                    .unwrap())
                };

                assert_eq!(lhs - rhs, C::Scalar::ZERO);
            }

            assert_eq!(phi[u], C::Scalar::ZERO);
        }

        let grand_sum_blind = Blind(C::Scalar::random(rng));
        let phi_commitment = params.commit_lagrange(&phi, grand_sum_blind).to_affine();

        // Hash grand sum commitment
        transcript.write_point(phi_commitment)?;

        end_timer!(lookup_commit_time);
        Ok(Committed {
            m_poly: pk.vk.domain.lagrange_to_coeff(self.m_values),
            phi_poly: pk.vk.domain.lagrange_to_coeff(phi),
        })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        pk: &ProvingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &pk.vk.domain;
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let phi_eval = eval_polynomial(&self.phi_poly, *x);
        let phi_next_eval = eval_polynomial(&self.phi_poly, x_next);
        let m_eval = eval_polynomial(&self.m_poly, *x);

        // Hash each advice evaluation
        for eval in iter::empty()
            .chain(Some(phi_eval))
            .chain(Some(phi_next_eval))
            .chain(Some(m_eval))
        {
            transcript.write_scalar(eval)?;
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.phi_poly,
                blind: Blind(C::Scalar::ZERO),
            }))
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.phi_poly,
                blind: Blind(C::Scalar::ZERO),
            }))
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.m_poly,
                blind: Blind(C::Scalar::ZERO),
            }))
    }
}

#[cfg(test)]
mod benches {
    use ark_std::rand::thread_rng;
    use ff::Field;
    use halo2curves::bn256::Fr;
    use std::collections::BTreeMap;
    use std::time::Instant;

    // bench the time to construct a BTreeMap out of a large table
    // tivm is short for table_index_value_mapping
    #[ignore]
    #[test]
    fn bench_tivm_btree_map() {
        env_logger::init();
        let mut rng = thread_rng();

        for log_n in 20..26 {
            let n = 1 << log_n;
            let dur = Instant::now();
            let _table: BTreeMap<Fr, usize> = (0..n)
                .map(|_| Fr::random(&mut rng))
                .enumerate()
                .map(|(i, x)| (x, i))
                .collect();
            log::info!(
                "construct btreemap from random vec (len = {}) took {:?}",
                n,
                dur.elapsed()
            );
        }

        for log_n in 20..26 {
            let n = 1 << log_n;
            let dur = Instant::now();
            let _table: BTreeMap<Fr, usize> = (0..n)
                .map(Fr::from)
                .enumerate()
                .map(|(i, x)| (x, i))
                .collect();
            log::info!(
                "construct btreemap from increasing vec (len = {}) took {:?}",
                n,
                dur.elapsed()
            );
        }
    }
}
