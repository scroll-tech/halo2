#![allow(clippy::many_single_char_names)]
#![allow(clippy::op_ref)]

// use assert_matches::assert_matches;
use halo2_proofs::arithmetic::{Field, FieldExt};
use halo2_proofs::circuit::{Cell, Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::{
    create_proof as create_plonk_proof, keygen_pk, keygen_vk, verify_proof as verify_plonk_proof,
    Advice, Assigned, Circuit, Column, ConstraintSystem, Error, Fixed, ProvingKey, TableColumn,
    VerifyingKey,
};
use halo2_proofs::poly::commitment::{CommitmentScheme, ParamsProver, Prover, Verifier};
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverGWC, VerifierGWC};
use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy;
use halo2_proofs::poly::Rotation;
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, TranscriptReadBuffer,
    TranscriptWriterBuffer,
};
use halo2curves::bn256::Bn256;
use rand_core::{OsRng, RngCore};
use std::marker::PhantomData;
use std::time::Instant;

const K: u32 = 17;

/// This represents an advice column at a certain row in the ConstraintSystem
#[derive(Copy, Clone, Debug)]
pub struct Variable(Column<Advice>, usize);

#[derive(Clone)]
struct PlonkConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    d: Column<Advice>,
    e: Column<Advice>,

    sa: Column<Fixed>,
    sb: Column<Fixed>,
    sc: Column<Fixed>,
    sm: Column<Fixed>,
    sp: Column<Fixed>,
    sl: TableColumn,
}

impl PlonkConfig {
    pub fn construct<F: FieldExt>(meta: &mut ConstraintSystem<F>) -> Self {
        let e = meta.advice_column();
        let a = meta.advice_column();
        let b = meta.advice_column();
        let sf = meta.fixed_column();
        let c = meta.advice_column();
        let d = meta.advice_column();
        let p = meta.instance_column();

        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);

        let sm = meta.fixed_column();
        let sa = meta.fixed_column();
        let sb = meta.fixed_column();
        let sc = meta.fixed_column();
        let sp = meta.fixed_column();
        let sl = meta.lookup_table_column();

        /*
         *   A         B      ...  sl
         * [
         *   instance  0      ...  0
         *   a         a      ...  0
         *   a         a^2    ...  0
         *   a         a      ...  0
         *   a         a^2    ...  0
         *   ...       ...    ...  ...
         *   ...       ...    ...  instance
         *   ...       ...    ...  a
         *   ...       ...    ...  a
         *   ...       ...    ...  0
         * ]
         */

        meta.lookup("lookup", |meta| {
            let a_ = meta.query_any(a, Rotation::cur());
            vec![(a_, sl)]
        });

        meta.create_gate("Combined add-mult", |meta| {
            let d = meta.query_advice(d, Rotation::next());
            let a = meta.query_advice(a, Rotation::cur());
            let sf = meta.query_fixed(sf, Rotation::cur());
            let e = meta.query_advice(e, Rotation::prev());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());

            let sa = meta.query_fixed(sa, Rotation::cur());
            let sb = meta.query_fixed(sb, Rotation::cur());
            let sc = meta.query_fixed(sc, Rotation::cur());
            let sm = meta.query_fixed(sm, Rotation::cur());

            vec![a.clone() * sa + b.clone() * sb + a * b * sm - (c * sc) + sf * (d * e)]
        });

        meta.create_gate("Public input", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let p = meta.query_instance(p, Rotation::cur());
            let sp = meta.query_fixed(sp, Rotation::cur());

            vec![sp * (a - p)]
        });

        meta.enable_equality(sf);
        meta.enable_equality(e);
        meta.enable_equality(d);
        meta.enable_equality(p);
        meta.enable_equality(sm);
        meta.enable_equality(sa);
        meta.enable_equality(sb);
        meta.enable_equality(sc);
        meta.enable_equality(sp);

        PlonkConfig {
            a,
            b,
            c,
            d,
            e,
            sa,
            sb,
            sc,
            sm,
            sp,
            sl,
        }
    }
}

#[allow(clippy::type_complexity)]
trait StandardCs<FF: FieldExt> {
    fn raw_multiply<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        f: F,
    ) -> Result<(Cell, Cell, Cell), Error>
    where
        F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>;
    fn raw_add<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        f: F,
    ) -> Result<(Cell, Cell, Cell), Error>
    where
        F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>;
    fn copy(&self, layouter: &mut impl Layouter<FF>, a: Cell, b: Cell) -> Result<(), Error>;
    fn public_input<F>(&self, layouter: &mut impl Layouter<FF>, f: F) -> Result<Cell, Error>
    where
        F: FnMut() -> Value<FF>;
    fn lookup_table(&self, layouter: &mut impl Layouter<FF>, values: &[FF]) -> Result<(), Error>;
}

struct StandardPlonk<F: FieldExt> {
    config: PlonkConfig,
    _marker: PhantomData<F>,
}

impl<FF: FieldExt> StandardPlonk<FF> {
    fn new(config: PlonkConfig) -> Self {
        StandardPlonk {
            config,
            _marker: PhantomData,
        }
    }
}

impl<FF: FieldExt> StandardCs<FF> for StandardPlonk<FF> {
    fn raw_multiply<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        mut f: F,
    ) -> Result<(Cell, Cell, Cell), Error>
    where
        F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>,
    {
        layouter.assign_region(
            || "raw_multiply",
            |mut region| {
                let mut value = None;
                let lhs = region.assign_advice(
                    || "lhs",
                    self.config.a,
                    0,
                    || {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    },
                )?;
                region.assign_advice(
                    || "lhs^4",
                    self.config.d,
                    0,
                    || value.unwrap().map(|v| v.0).square().square(),
                )?;
                let rhs = region.assign_advice(
                    || "rhs",
                    self.config.b,
                    0,
                    || value.unwrap().map(|v| v.1),
                )?;
                region.assign_advice(
                    || "rhs^4",
                    self.config.e,
                    0,
                    || value.unwrap().map(|v| v.1).square().square(),
                )?;
                let out = region.assign_advice(
                    || "out",
                    self.config.c,
                    0,
                    || value.unwrap().map(|v| v.2),
                )?;

                region.assign_fixed(|| "a", self.config.sa, 0, || Value::known(FF::zero()))?;
                region.assign_fixed(|| "b", self.config.sb, 0, || Value::known(FF::zero()))?;
                region.assign_fixed(|| "c", self.config.sc, 0, || Value::known(FF::one()))?;
                region.assign_fixed(|| "a * b", self.config.sm, 0, || Value::known(FF::one()))?;
                Ok((lhs.cell(), rhs.cell(), out.cell()))
            },
        )
    }
    fn raw_add<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        mut f: F,
    ) -> Result<(Cell, Cell, Cell), Error>
    where
        F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>,
    {
        layouter.assign_region(
            || "raw_add",
            |mut region| {
                let mut value = None;
                let lhs = region.assign_advice(
                    || "lhs",
                    self.config.a,
                    0,
                    || {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    },
                )?;
                region.assign_advice(
                    || "lhs^4",
                    self.config.d,
                    0,
                    || value.unwrap().map(|v| v.0).square().square(),
                )?;
                let rhs = region.assign_advice(
                    || "rhs",
                    self.config.b,
                    0,
                    || value.unwrap().map(|v| v.1),
                )?;
                region.assign_advice(
                    || "rhs^4",
                    self.config.e,
                    0,
                    || value.unwrap().map(|v| v.1).square().square(),
                )?;
                let out = region.assign_advice(
                    || "out",
                    self.config.c,
                    0,
                    || value.unwrap().map(|v| v.2),
                )?;

                region.assign_fixed(|| "a", self.config.sa, 0, || Value::known(FF::one()))?;
                region.assign_fixed(|| "b", self.config.sb, 0, || Value::known(FF::one()))?;
                region.assign_fixed(|| "c", self.config.sc, 0, || Value::known(FF::one()))?;
                region.assign_fixed(|| "a * b", self.config.sm, 0, || Value::known(FF::zero()))?;
                Ok((lhs.cell(), rhs.cell(), out.cell()))
            },
        )
    }
    fn copy(&self, layouter: &mut impl Layouter<FF>, left: Cell, right: Cell) -> Result<(), Error> {
        layouter.assign_region(
            || "copy",
            |mut region| {
                region.constrain_equal(left, right)?;
                region.constrain_equal(left, right)
            },
        )
    }
    fn public_input<F>(&self, layouter: &mut impl Layouter<FF>, mut f: F) -> Result<Cell, Error>
    where
        F: FnMut() -> Value<FF>,
    {
        layouter.assign_region(
            || "public_input",
            |mut region| {
                let value = region.assign_advice(|| "value", self.config.a, 0, &mut f)?;
                region.assign_fixed(|| "public", self.config.sp, 0, || Value::known(FF::one()))?;

                Ok(value.cell())
            },
        )
    }
    fn lookup_table(&self, layouter: &mut impl Layouter<FF>, values: &[FF]) -> Result<(), Error> {
        layouter.assign_table(
            || "",
            |mut table| {
                for (index, &value) in values.iter().enumerate() {
                    table.assign_cell(
                        || "table col",
                        self.config.sl,
                        index,
                        || Value::known(value),
                    )?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }
}

macro_rules! common {
    ($scheme:ident) => {{
        let a = <$scheme as CommitmentScheme>::Scalar::from(2834758237)
            * <$scheme as CommitmentScheme>::Scalar::ZETA;
        let instance = <$scheme as CommitmentScheme>::Scalar::one()
            + <$scheme as CommitmentScheme>::Scalar::one();
        let lookup_table = vec![
            instance,
            a,
            a,
            <$scheme as CommitmentScheme>::Scalar::zero(),
        ];
        (a, instance, lookup_table)
    }};
}

fn verify_proof<
    'a,
    'params,
    Scheme: CommitmentScheme,
    V: Verifier<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    T: TranscriptReadBuffer<&'a [u8], Scheme::Curve, E>,
    Strategy: VerificationStrategy<'params, Scheme, V, Output = Strategy>,
>(
    params_verifier: &'params Scheme::ParamsVerifier,
    vk: &VerifyingKey<Scheme::Curve>,
    proof: &'a [u8],
) {
    let (_, instance, _) = common!(Scheme);
    let pubinputs = vec![instance];

    let mut transcript = T::init(proof);

    let strategy = Strategy::new(params_verifier);
    let strategy = verify_plonk_proof(
        params_verifier,
        vk,
        strategy,
        &[&[&pubinputs[..]], &[&pubinputs[..]]],
        &mut transcript,
    )
    .unwrap();

    assert!(strategy.finalize());
}

#[test]
fn plonk_api() {
    #[derive(Clone)]
    struct MyCircuit<F: FieldExt> {
        a: Value<F>,
        lookup_table: Vec<F>,
    }

    impl<F: FieldExt> Circuit<F> for MyCircuit<F> {
        type Config = PlonkConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                a: Value::unknown(),
                lookup_table: self.lookup_table.clone(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> PlonkConfig {
            PlonkConfig::construct(meta)
        }

        fn synthesize(
            &self,
            config: PlonkConfig,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let cs = StandardPlonk::new(config);
            let mut is_first_pass_vec = vec![true; 8];

            let _ = cs.public_input(&mut layouter, || Value::known(F::one() + F::one()))?;

            let a: Value<Assigned<_>> = self.a.into();
            let parallel_regions_time = Instant::now();
            #[cfg(feature = "parallel_syn")]
            layouter.assign_regions(
                || "regions",
                (0..8)
                    .into_iter()
                    .zip(is_first_pass_vec.chunks_mut(1).into_iter())
                    .map(|(_, is_first_pass)| {
                        |mut region: Region<'_, F>| -> Result<(), Error> {
                            let n = 1 << 13;
                            for i in 0..n {
                                // skip the assign of rows except the last row in the first pass
                                if is_first_pass[0] && i < n - 1 {
                                    continue;
                                }
                                let a0 =
                                    region.assign_advice(|| "config.a", cs.config.a, i, || a)?;
                                let a1 =
                                    region.assign_advice(|| "config.b", cs.config.b, i, || a)?;
                                region.assign_advice(
                                    || "config.c",
                                    cs.config.c,
                                    i,
                                    || a.double(),
                                )?;

                                region.assign_fixed(
                                    || "a",
                                    cs.config.sa,
                                    i,
                                    || Value::known(F::one()),
                                )?;
                                region.assign_fixed(
                                    || "b",
                                    cs.config.sb,
                                    i,
                                    || Value::known(F::one()),
                                )?;
                                region.assign_fixed(
                                    || "c",
                                    cs.config.sc,
                                    i,
                                    || Value::known(F::one()),
                                )?;
                                region.assign_fixed(
                                    || "a * b",
                                    cs.config.sm,
                                    i,
                                    || Value::known(F::zero()),
                                )?;

                                region.constrain_equal(a0.cell(), a1.cell())?;
                            }
                            is_first_pass[0] = false;
                            Ok(())
                        }
                    })
                    .collect(),
            )?;
            log::info!(
                "parallel_regions assign took {:?}",
                parallel_regions_time.elapsed()
            );

            for _ in 0..10 {
                let a: Value<Assigned<_>> = self.a.into();
                let mut a_squared = Value::unknown();
                let (a0, _, c0) = cs.raw_multiply(&mut layouter, || {
                    a_squared = a.square();
                    a.zip(a_squared).map(|(a, a_squared)| (a, a, a_squared))
                })?;
                let (a1, b1, _) = cs.raw_add(&mut layouter, || {
                    let fin = a_squared + a;
                    a.zip(a_squared)
                        .zip(fin)
                        .map(|((a, a_squared), fin)| (a, a_squared, fin))
                })?;
                cs.copy(&mut layouter, a0, a1)?;
                cs.copy(&mut layouter, b1, c0)?;
            }

            cs.lookup_table(&mut layouter, &self.lookup_table)?;

            Ok(())
        }
    }
    fn keygen<Scheme: CommitmentScheme>(
        params: &Scheme::ParamsProver,
    ) -> ProvingKey<Scheme::Curve> {
        let (_, _, lookup_table) = common!(Scheme);
        let empty_circuit: MyCircuit<Scheme::Scalar> = MyCircuit {
            a: Value::unknown(),
            lookup_table,
        };

        // Initialize the proving key
        let vk = keygen_vk(params, &empty_circuit).expect("keygen_vk should not fail");
        log::info!("keygen vk succeed");

        let pk = keygen_pk(params, vk, &empty_circuit).expect("keygen_pk should not fail");
        log::info!("keygen pk succeed");

        pk
    }

    fn create_proof<
        'params,
        Scheme: CommitmentScheme,
        P: Prover<'params, Scheme>,
        E: EncodedChallenge<Scheme::Curve>,
        R: RngCore,
        T: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
    >(
        rng: R,
        params: &'params Scheme::ParamsProver,
        pk: &ProvingKey<Scheme::Curve>,
    ) -> Vec<u8> {
        let (a, instance, lookup_table) = common!(Scheme);

        let circuit: MyCircuit<Scheme::Scalar> = MyCircuit {
            a: Value::known(a),
            lookup_table,
        };

        let mut transcript = T::init(vec![]);

        create_plonk_proof::<Scheme, P, _, _, _, _>(
            params,
            pk,
            &[circuit.clone(), circuit],
            &[&[&[instance]], &[&[instance]]],
            rng,
            &mut transcript,
        )
        .expect("proof generation should not fail");

        transcript.finalize()
    }

    fn test_plonk_api_gwc() {
        use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
        use halo2_proofs::poly::kzg::multiopen::{ProverGWC, VerifierGWC};
        use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy;
        use halo2curves::bn256::Bn256;

        type Scheme = KZGCommitmentScheme<Bn256>;
        // bad_keys!(Scheme);

        let (a, instance, lookup_table) = common!(Scheme);

        let circuit: MyCircuit<<Scheme as CommitmentScheme>::Scalar> = MyCircuit {
            a: Value::known(a),
            lookup_table,
        };

        // Check this circuit is satisfied.
        let prover = match MockProver::run(K, &circuit, vec![vec![instance]]) {
            Ok(prover) => prover,
            Err(e) => panic!("{:?}", e),
        };
        assert_eq!(prover.verify_par(), Ok(()));
        log::info!("mock proving succeed!");

        let params = ParamsKZG::<Bn256>::new(K);
        let rng = OsRng;

        let pk = keygen::<KZGCommitmentScheme<_>>(&params);

        let proof = create_proof::<_, ProverGWC<_>, _, _, Blake2bWrite<_, _, Challenge255<_>>>(
            rng, &params, &pk,
        );

        let verifier_params = params.verifier_params();

        verify_proof::<
            _,
            VerifierGWC<_>,
            _,
            Blake2bRead<_, _, Challenge255<_>>,
            AccumulatorStrategy<_>,
        >(verifier_params, pk.get_vk(), &proof[..]);
    }

    fn test_plonk_api_shplonk() {
        use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
        use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
        use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy;
        use halo2curves::bn256::Bn256;

        // type Scheme = KZGCommitmentScheme<Bn256>;
        // bad_keys!(Scheme);

        let params = ParamsKZG::<Bn256>::new(K);
        let rng = OsRng;

        let pk = keygen::<KZGCommitmentScheme<_>>(&params);

        let proof = create_proof::<_, ProverSHPLONK<_>, _, _, Blake2bWrite<_, _, Challenge255<_>>>(
            rng, &params, &pk,
        );

        let verifier_params = params.verifier_params();

        verify_proof::<
            _,
            VerifierSHPLONK<_>,
            _,
            Blake2bRead<_, _, Challenge255<_>>,
            AccumulatorStrategy<_>,
        >(verifier_params, pk.get_vk(), &proof[..]);
    }

    fn test_plonk_api_ipa() {
        use halo2_proofs::poly::ipa::commitment::{IPACommitmentScheme, ParamsIPA};
        use halo2_proofs::poly::ipa::multiopen::{ProverIPA, VerifierIPA};
        use halo2_proofs::poly::ipa::strategy::AccumulatorStrategy;
        use halo2curves::pasta::EqAffine;

        // type Scheme = IPACommitmentScheme<EqAffine>;
        // bad_keys!(Scheme);

        let params = ParamsIPA::<EqAffine>::new(K);
        let rng = OsRng;

        let pk = keygen::<IPACommitmentScheme<EqAffine>>(&params);

        let proof = create_proof::<_, ProverIPA<_>, _, _, Blake2bWrite<_, _, Challenge255<_>>>(
            rng, &params, &pk,
        );

        let verifier_params = params.verifier_params();

        verify_proof::<
            _,
            VerifierIPA<_>,
            _,
            Blake2bRead<_, _, Challenge255<_>>,
            AccumulatorStrategy<_>,
        >(verifier_params, pk.get_vk(), &proof[..]);
    }

    env_logger::init();
    test_plonk_api_ipa();
    test_plonk_api_gwc();
    test_plonk_api_shplonk();
}

#[test]
fn plonk_api_with_many_subregions() {
    #[derive(Clone)]
    struct MyCircuit<F: FieldExt> {
        a: Value<F>,
        lookup_table: Vec<F>,
    }

    impl<F: FieldExt> Circuit<F> for MyCircuit<F> {
        type Config = PlonkConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                a: Value::unknown(),
                lookup_table: self.lookup_table.clone(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> PlonkConfig {
            PlonkConfig::construct(meta)
        }

        fn synthesize(
            &self,
            config: PlonkConfig,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let cs = StandardPlonk::new(config);

            let _ = cs.public_input(&mut layouter, || Value::known(F::one() + F::one()))?;

            let a: Value<Assigned<_>> = self.a.into();
            let parallel_regions_time = Instant::now();
            #[cfg(feature = "parallel_syn")]
            layouter.assign_regions(
                || "regions",
                (0..(1 << 15))
                    .into_iter()
                    .map(|_| {
                        let mut is_first_pass = true;
                        move |mut region: Region<'_, F>| -> Result<(), Error> {
                            let n = 1 << 1;
                            for i in 0..n {
                                // skip the assign of rows except the last row in the first pass
                                if is_first_pass && i < n - 1 {
                                    continue;
                                }
                                let a0 =
                                    region.assign_advice(|| "config.a", cs.config.a, i, || a)?;
                                let a1 =
                                    region.assign_advice(|| "config.b", cs.config.b, i, || a)?;
                                region.assign_advice(
                                    || "config.c",
                                    cs.config.c,
                                    i,
                                    || a.double(),
                                )?;

                                region.assign_fixed(
                                    || "a",
                                    cs.config.sa,
                                    i,
                                    || Value::known(F::one()),
                                )?;
                                region.assign_fixed(
                                    || "b",
                                    cs.config.sb,
                                    i,
                                    || Value::known(F::one()),
                                )?;
                                region.assign_fixed(
                                    || "c",
                                    cs.config.sc,
                                    i,
                                    || Value::known(F::one()),
                                )?;
                                region.assign_fixed(
                                    || "a * b",
                                    cs.config.sm,
                                    i,
                                    || Value::known(F::zero()),
                                )?;

                                region.constrain_equal(a0.cell(), a1.cell())?;
                            }
                            is_first_pass = false;
                            Ok(())
                        }
                    })
                    .collect(),
            )?;
            log::info!(
                "parallel_regions assign took {:?}",
                parallel_regions_time.elapsed()
            );

            for _ in 0..10 {
                let a: Value<Assigned<_>> = self.a.into();
                let mut a_squared = Value::unknown();
                let (a0, _, c0) = cs.raw_multiply(&mut layouter, || {
                    a_squared = a.square();
                    a.zip(a_squared).map(|(a, a_squared)| (a, a, a_squared))
                })?;
                let (a1, b1, _) = cs.raw_add(&mut layouter, || {
                    let fin = a_squared + a;
                    a.zip(a_squared)
                        .zip(fin)
                        .map(|((a, a_squared), fin)| (a, a_squared, fin))
                })?;
                cs.copy(&mut layouter, a0, a1)?;
                cs.copy(&mut layouter, b1, c0)?;
            }

            cs.lookup_table(&mut layouter, &self.lookup_table)?;

            Ok(())
        }
    }
    fn keygen<Scheme: CommitmentScheme>(
        params: &Scheme::ParamsProver,
    ) -> ProvingKey<Scheme::Curve> {
        let (_, _, lookup_table) = common!(Scheme);
        let empty_circuit: MyCircuit<Scheme::Scalar> = MyCircuit {
            a: Value::unknown(),
            lookup_table,
        };

        // Initialize the proving key
        let vk = keygen_vk(params, &empty_circuit).expect("keygen_vk should not fail");
        log::info!("keygen vk succeed");

        let pk = keygen_pk(params, vk, &empty_circuit).expect("keygen_pk should not fail");
        log::info!("keygen pk succeed");

        pk
    }

    fn create_proof<
        'params,
        Scheme: CommitmentScheme,
        P: Prover<'params, Scheme>,
        E: EncodedChallenge<Scheme::Curve>,
        R: RngCore,
        T: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
    >(
        rng: R,
        params: &'params Scheme::ParamsProver,
        pk: &ProvingKey<Scheme::Curve>,
    ) -> Vec<u8> {
        let (a, instance, lookup_table) = common!(Scheme);

        let circuit: MyCircuit<Scheme::Scalar> = MyCircuit {
            a: Value::known(a),
            lookup_table,
        };

        let mut transcript = T::init(vec![]);

        create_plonk_proof::<Scheme, P, _, _, _, _>(
            params,
            pk,
            &[circuit.clone(), circuit],
            &[&[&[instance]], &[&[instance]]],
            rng,
            &mut transcript,
        )
        .expect("proof generation should not fail");

        transcript.finalize()
    }

    type Scheme = KZGCommitmentScheme<Bn256>;
    // bad_keys!(Scheme);

    let (a, instance, lookup_table) = common!(Scheme);

    let circuit: MyCircuit<<Scheme as CommitmentScheme>::Scalar> = MyCircuit {
        a: Value::known(a),
        lookup_table,
    };

    // Check this circuit is satisfied.
    let prover = match MockProver::run(K, &circuit, vec![vec![instance]]) {
        Ok(prover) => prover,
        Err(e) => panic!("{:?}", e),
    };
    assert_eq!(prover.verify_par(), Ok(()));
    log::info!("mock proving succeed!");

    let params = ParamsKZG::<Bn256>::new(K);
    let rng = OsRng;

    let pk = keygen::<KZGCommitmentScheme<_>>(&params);

    let proof = create_proof::<_, ProverGWC<_>, _, _, Blake2bWrite<_, _, Challenge255<_>>>(
        rng, &params, &pk,
    );

    let verifier_params = params.verifier_params();

    verify_proof::<_, VerifierGWC<_>, _, Blake2bRead<_, _, Challenge255<_>>, AccumulatorStrategy<_>>(
        verifier_params,
        pk.get_vk(),
        &proof[..],
    );
}
