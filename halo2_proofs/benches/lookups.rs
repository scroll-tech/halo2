#[macro_use]
extern crate criterion;

use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk::*;
use halo2_proofs::poly::kzg::multiopen::VerifierGWC;
use halo2_proofs::poly::{commitment::ParamsProver, Rotation};
use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
use halo2curves::bn256::{Bn256, G1Affine};
use halo2curves::pairing::Engine;
use rand_core::OsRng;

use halo2_proofs::{
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::ProverGWC,
        strategy::SingleStrategy,
    },
    transcript::{TranscriptReadBuffer, TranscriptWriterBuffer},
};

use std::marker::PhantomData;

use criterion::{BenchmarkId, Criterion};
use ff::PrimeField as Field;

fn criterion_benchmark(c: &mut Criterion) {
    #[derive(Clone, Default)]
    struct MyCircuit<F: Field> {
        _marker: PhantomData<F>,
    }

    #[derive(Clone)]
    struct MyConfig {
        selector: Selector,
        table: TableColumn,
        advice: Column<Advice>,
        other_advice: Column<Advice>,
    }

    impl<F: Field> Circuit<F> for MyCircuit<F> {
        type Config = MyConfig;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> MyConfig {
            let config = MyConfig {
                selector: meta.complex_selector(),
                table: meta.lookup_table_column(),
                advice: meta.advice_column(),
                other_advice: meta.advice_column(),
            };

            let dummy_selector = meta.complex_selector();

            meta.create_gate("degree 6 gate", |meta| {
                let dummy_selector = meta.query_selector(dummy_selector);
                let constraints = std::iter::repeat(dummy_selector.clone())
                    .take(4)
                    .fold(dummy_selector.clone(), |acc, val| acc * val.clone());
                Constraints::with_selector(dummy_selector, Some(constraints))
            });

            meta.lookup("lookup", |meta| {
                let advice = meta.query_advice(config.advice, Rotation::cur());
                vec![(advice, config.table)]
            });

            meta.lookup("lookup", |meta| {
                let advice = meta.query_advice(config.advice, Rotation::cur());
                vec![(advice, config.table)]
            });

            meta.lookup("lookup", |meta| {
                let advice = meta.query_advice(config.advice, Rotation::cur());
                vec![(advice, config.table)]
            });

            meta.lookup("lookup", |meta| {
                let advice = meta.query_advice(config.advice, Rotation::cur());
                vec![(advice, config.table)]
            });

            meta.lookup("lookup", |meta| {
                let advice = meta.query_advice(config.advice, Rotation::cur());
                vec![(advice, config.table)]
            });

            /*
                - We need degree at least 6 because 6 - 1 = 5 and we need to go to extended domain of 8n
                - Our goal is to get to max degree of 9 because now 9 - 1 = 8 and that will fit into domain

                - base degree = table_deg + 2
                - if we put input_expression_degree = 1
                => degree = base + 1 = 3 + 1 = 4
                - we can batch one more with 5 more lookups
            */

            config
        }

        fn synthesize(
            &self,
            config: MyConfig,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter.assign_table(
                || "8-bit table",
                |mut table| {
                    for row in 0u64..(1 << 8) {
                        table.assign_cell(
                            || format!("row {}", row),
                            config.table,
                            row as usize,
                            || Value::known(F::from(row)),
                        )?;
                    }

                    Ok(())
                },
            )?;

            layouter.assign_region(
                || "assign values",
                |mut region| {
                    for offset in 0u64..(1 << 10) {
                        config.selector.enable(&mut region, offset as usize)?;
                        region.assign_advice(
                            || format!("offset {}", offset),
                            config.advice,
                            offset as usize,
                            || Value::known(F::from(offset % 256)),
                        )?;
                    }
                    for offset in 1u64..(1 << 10) {
                        config.selector.enable(&mut region, offset as usize)?;
                        region.assign_advice(
                            || format!("offset {}", offset),
                            config.other_advice,
                            offset as usize - 1,
                            || Value::known(F::from(offset % 256)),
                        )?;
                    }
                    Ok(())
                },
            )
        }
    }

    fn keygen(k: u32) -> (ParamsKZG<Bn256>, ProvingKey<G1Affine>) {
        let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
        let empty_circuit: MyCircuit<<Bn256 as Engine>::Scalar> = MyCircuit {
            _marker: PhantomData,
        };
        let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
        let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");
        (params, pk)
    }

    fn prover(_k: u32, params: &ParamsKZG<Bn256>, pk: &ProvingKey<G1Affine>) -> Vec<u8> {
        let rng = OsRng;

        let circuit: MyCircuit<<Bn256 as Engine>::Scalar> = MyCircuit {
            _marker: PhantomData,
        };

        let mut transcript = Blake2bWrite::<_, _, Challenge255<G1Affine>>::init(vec![]);
        create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<'_, Bn256>, _, _, _, _>(
            params,
            pk,
            &[circuit],
            &[&[]],
            rng,
            &mut transcript,
        )
        .expect("proof generation should not fail");
        transcript.finalize()
    }

    fn verifier(params: &ParamsKZG<Bn256>, vk: &VerifyingKey<G1Affine>, proof: &[u8]) {
        let strategy = SingleStrategy::new(params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<G1Affine>>::init(proof);
        assert!(verify_proof::<
            KZGCommitmentScheme<Bn256>,
            VerifierGWC<'_, Bn256>,
            Challenge255<G1Affine>,
            Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
            SingleStrategy<'_, Bn256>,
        >(params, vk, strategy, &[&[]], &mut transcript)
        .is_ok());
    }

    let k_range = 16..=16;

    let mut keygen_group = c.benchmark_group("plonk-keygen");
    keygen_group.sample_size(10);
    for k in k_range.clone() {
        keygen_group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| keygen(k));
        });
    }
    keygen_group.finish();

    let mut prover_group = c.benchmark_group("plonk-prover");
    prover_group.sample_size(10);
    for k in k_range.clone() {
        let (params, pk) = keygen(k);

        prover_group.bench_with_input(
            BenchmarkId::from_parameter(k),
            &(k, &params, &pk),
            |b, &(k, params, pk)| {
                b.iter(|| prover(k, params, pk));
            },
        );
    }
    prover_group.finish();

    let mut verifier_group = c.benchmark_group("plonk-verifier");
    for k in k_range {
        let (params, pk) = keygen(k);
        let proof = prover(k, &params, &pk);

        verifier_group.bench_with_input(
            BenchmarkId::from_parameter(k),
            &(&params, pk.get_vk(), &proof[..]),
            |b, &(params, vk, proof)| {
                b.iter(|| verifier(params, vk, proof));
            },
        );
    }
    verifier_group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
