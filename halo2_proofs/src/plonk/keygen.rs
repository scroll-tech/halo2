#![allow(clippy::int_plus_one)]

use std::ops::Range;
use std::sync::Arc;

use ff::{Field, FromUniformBytes};
use group::Curve;

use super::{
    circuit::{
        Advice, Any, Assignment, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner, Instance,
        Selector,
    },
    evaluation::Evaluator,
    permutation, Assigned, Challenge, Error, LagrangeCoeff, Polynomial, ProvingKey, VerifyingKey,
};
use crate::helpers::CopyCell;
use crate::{
    arithmetic::{parallelize, CurveAffine},
    circuit::Value,
    poly::{
        batch_invert_assigned,
        commitment::{Blind, Params},
        EvaluationDomain,
    },
    two_dim_vec_to_vec_of_slice,
};

pub(crate) fn create_domain<C, ConcreteCircuit>(
    k: u32,
    #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
) -> (
    EvaluationDomain<C::Scalar>,
    ConstraintSystem<C::Scalar>,
    ConcreteCircuit::Config,
)
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let mut cs = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut cs, params);
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut cs);

    let cs = cs.chunk_lookups();

    let degree = cs.degree();

    let domain = EvaluationDomain::new(degree as u32, k);

    (domain, cs, config)
}

/// Assembly to be used in circuit synthesis.
#[derive(Debug)]
struct Assembly<'a, F: Field> {
    k: u32,
    fixed_vec: Arc<Vec<Polynomial<Assigned<F>, LagrangeCoeff>>>,
    fixed: Vec<&'a mut [Assigned<F>]>,
    permutation: Option<permutation::keygen::Assembly>,
    selectors_vec: Arc<Vec<Vec<bool>>>,
    selectors: Vec<&'a mut [bool]>,
    rw_rows: Range<usize>,
    copies: Vec<(CopyCell, CopyCell)>,
    // A range of available rows for assignment and copies.
    usable_rows: Range<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<'a, F: Field> Assignment<F> for Assembly<'a, F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&mut self) {
        // Do nothing; we don't care about regions in this context.
    }

    fn enable_selector<A, AR>(&mut self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        if !self.rw_rows.contains(&row) {
            log::error!("enable_selector: {:?}, row: {}", selector, row);
            return Err(Error::Synthesis);
        }

        self.selectors[selector.0][row - self.rw_rows.start] = true;

        Ok(())
    }

    fn fork(&mut self, ranges: &[Range<usize>]) -> Result<Vec<Self>, Error> {
        let mut range_start = self.rw_rows.start;
        for (i, sub_range) in ranges.iter().enumerate() {
            if sub_range.start < range_start {
                // TODO: use more precise error type
                log::error!(
                    "subCS_{} sub_range.start ({}) < range_start ({})",
                    i,
                    sub_range.start,
                    range_start
                );
                return Err(Error::Synthesis);
            }
            if i == ranges.len() - 1 && sub_range.end > self.rw_rows.end {
                log::error!(
                    "subCS_{} sub_range.end ({}) > self.rw_rows.end ({})",
                    i,
                    sub_range.end,
                    self.rw_rows.end
                );
                return Err(Error::Synthesis);
            }
            range_start = sub_range.end;
            log::debug!(
                "subCS_{} rw_rows: {}..{}",
                i,
                sub_range.start,
                sub_range.end
            );
        }

        let fixed_ptrs = self
            .fixed
            .iter_mut()
            .map(|vec| vec.as_mut_ptr())
            .collect::<Vec<_>>();
        let selectors_ptrs = self
            .selectors
            .iter_mut()
            .map(|vec| vec.as_mut_ptr())
            .collect::<Vec<_>>();

        let mut sub_cs = vec![];
        for sub_range in ranges {
            let fixed = fixed_ptrs
                .iter()
                .map(|ptr| unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(sub_range.start),
                        sub_range.end - sub_range.start,
                    )
                })
                .collect::<Vec<&mut [Assigned<F>]>>();
            let selectors = selectors_ptrs
                .iter()
                .map(|ptr| unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.add(sub_range.start),
                        sub_range.end - sub_range.start,
                    )
                })
                .collect::<Vec<&mut [bool]>>();

            sub_cs.push(Self {
                k: 0,
                fixed_vec: self.fixed_vec.clone(),
                fixed,
                permutation: None,
                selectors_vec: self.selectors_vec.clone(),
                selectors,
                rw_rows: sub_range.clone(),
                copies: vec![],
                usable_rows: self.usable_rows.clone(),
                _marker: Default::default(),
            });
        }

        Ok(sub_cs)
    }

    fn merge(&mut self, sub_cs: Vec<Self>) -> Result<(), Error> {
        for (left, right) in sub_cs.into_iter().flat_map(|cs| cs.copies.into_iter()) {
            self.permutation
                .as_mut()
                .expect("permutation must be Some")
                .copy(left.column, left.row, right.column, right.row)?;
        }
        Ok(())
    }

    fn query_advice(&self, _column: Column<Advice>, _row: usize) -> Result<F, Error> {
        // We only care about fixed columns here
        Ok(F::ZERO)
    }

    fn query_fixed(&self, column: Column<Fixed>, row: usize) -> Result<F, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }
        if !self.rw_rows.contains(&row) {
            log::error!("query_fixed: {:?}, row: {}", column, row);
            return Err(Error::Synthesis);
        }
        self.fixed
            .get(column.index())
            .and_then(|v| v.get(row - self.rw_rows.start))
            .map(|v| v.evaluate())
            .ok_or(Error::BoundsFailure)
    }

    fn query_instance(&self, _: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        // There is no instance in this context.
        Ok(Value::unknown())
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about fixed columns here
        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Fixed>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        if !self.rw_rows.contains(&row) {
            log::error!("assign_fixed: {:?}, row: {}", column, row);
            return Err(Error::Synthesis);
        }

        *self
            .fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row - self.rw_rows.start))
            .expect("bounds failure") = to().into_field().assign()?;

        Ok(())
    }

    fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        match self.permutation.as_mut() {
            None => {
                self.copies.push((
                    CopyCell {
                        column: left_column,
                        row: left_row,
                    },
                    CopyCell {
                        column: right_column,
                        row: right_row,
                    },
                ));
                Ok(())
            }
            Some(permutation) => permutation.copy(left_column, left_row, right_column, right_row),
        }
    }

    fn fill_from_row(
        &mut self,
        column: Column<Fixed>,
        from_row: usize,
        to: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&from_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let col = self.fixed.get_mut(column.index()).expect("bounds failure");

        let filler = to.assign()?;
        for row in self.usable_rows.clone().skip(from_row) {
            col[row] = filler;
        }

        Ok(())
    }

    fn get_challenge(&self, _: Challenge) -> Value<F> {
        Value::unknown()
    }

    fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // Do nothing
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self, _: Option<String>) {
        // Do nothing; we don't care about namespaces in this context.
    }
}

/// Generate a `VerifyingKey` from an instance of `Circuit`.
pub fn keygen_vk<'params, C, P, ConcreteCircuit>(
    params: &P,
    circuit: &ConcreteCircuit,
) -> Result<VerifyingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    let (domain, cs, config) = create_domain::<C, ConcreteCircuit>(
        params.k(),
        #[cfg(feature = "circuit-params")]
        circuit.params(),
    );

    if (params.n() as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k()));
    }

    let fixed_vec = Arc::new(vec![domain.empty_lagrange_assigned(); cs.num_fixed_columns]);
    let fixed = unsafe {
        let fixed_vec_clone = fixed_vec.clone();
        let ptr = Arc::as_ptr(&fixed_vec_clone) as *mut Vec<Polynomial<Assigned<_>, LagrangeCoeff>>;
        let mut_ref = &mut (*ptr);
        mut_ref
            .iter_mut()
            .map(|poly| poly.values.as_mut_slice())
            .collect::<Vec<_>>()
    };

    let selectors_vec = Arc::new(vec![vec![false; params.n() as usize]; cs.num_selectors]);
    let selectors = unsafe {
        let selectors_vec_clone = selectors_vec.clone();
        let ptr = Arc::as_ptr(&selectors_vec_clone) as *mut Vec<Vec<bool>>;
        let mut_ref = &mut (*ptr);
        mut_ref
            .iter_mut()
            .map(|vec| vec.as_mut_slice())
            .collect::<Vec<_>>()
    };
    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k(),
        fixed_vec,
        fixed,
        permutation: Some(permutation::keygen::Assembly::new(
            params.n() as usize,
            &cs.permutation,
        )),
        selectors_vec,
        selectors,
        copies: vec![],
        rw_rows: 0..params.n() as usize - (cs.blinding_factors() + 1),
        usable_rows: 0..params.n() as usize - (cs.blinding_factors() + 1),
        _marker: std::marker::PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    debug_assert_eq!(Arc::strong_count(&assembly.fixed_vec), 1);
    debug_assert_eq!(Arc::strong_count(&assembly.selectors_vec), 1);
    let mut fixed =
        batch_invert_assigned(Arc::try_unwrap(assembly.fixed_vec).expect("only one Arc for fixed"));
    let (cs, selector_polys) = cs.compress_selectors(
        Arc::try_unwrap(assembly.selectors_vec).expect("only one Arc for selectors"),
    );
    fixed.extend(
        selector_polys
            .into_iter()
            .map(|poly| domain.lagrange_from_vec(poly)),
    );

    let permutation_vk = assembly
        .permutation
        .take()
        .expect("permutation must be Some")
        .build_vk(params, &domain, &cs.permutation);

    let fixed_commitments = fixed
        .iter()
        .map(|poly| params.commit_lagrange(poly, Blind::default()).to_affine())
        .collect();

    Ok(VerifyingKey::from_parts(
        domain,
        fixed_commitments,
        permutation_vk,
        cs,
        //        assembly.selectors,
    ))
}

/// Generate a `ProvingKey` from an instance of `Circuit`.
pub fn keygen_pk2<'params, C, P, ConcreteCircuit>(
    params: &P,
    circuit: &ConcreteCircuit,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    keygen_pk_impl(params, None, circuit)
}

/// Generate a `ProvingKey` from a `VerifyingKey` and an instance of `Circuit`
pub fn keygen_pk<'params, C, P, ConcreteCircuit>(
    params: &P,
    vk: VerifyingKey<C>,
    circuit: &ConcreteCircuit,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    keygen_pk_impl(params, Some(vk), circuit)
}

/// Generate a `ProvingKey` from a `VerifyingKey` and an instance of `Circuit`.
pub fn keygen_pk_impl<'params, C, P, ConcreteCircuit>(
    params: &P,
    vk: Option<VerifyingKey<C>>,
    circuit: &ConcreteCircuit,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    let (domain, cs, config) = create_domain::<C, ConcreteCircuit>(
        params.k(),
        #[cfg(feature = "circuit-params")]
        circuit.params(),
    );

    if (params.n() as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k()));
    }

    let fixed_vec = Arc::new(vec![domain.empty_lagrange_assigned(); cs.num_fixed_columns]);
    let fixed = two_dim_vec_to_vec_of_slice!(fixed_vec);

    let selectors_vec = Arc::new(vec![vec![false; params.n() as usize]; cs.num_selectors]);
    let selectors = two_dim_vec_to_vec_of_slice!(selectors_vec);

    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k(),
        fixed_vec,
        fixed,
        permutation: Some(permutation::keygen::Assembly::new(
            params.n() as usize,
            &cs.permutation,
        )),
        selectors_vec,
        selectors,
        copies: vec![],
        rw_rows: 0..params.n() as usize - (cs.blinding_factors() + 1),
        usable_rows: 0..params.n() as usize - (cs.blinding_factors() + 1),
        _marker: std::marker::PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    debug_assert_eq!(Arc::strong_count(&assembly.fixed_vec), 1);
    debug_assert_eq!(Arc::strong_count(&assembly.selectors_vec), 1);
    let mut fixed =
        batch_invert_assigned(Arc::try_unwrap(assembly.fixed_vec).expect("only one Arc for fixed"));
    let (cs, selector_polys) = cs.compress_selectors(
        Arc::try_unwrap(assembly.selectors_vec).expect("only one Arc for selectors"),
    );
    fixed.extend(
        selector_polys
            .into_iter()
            .map(|poly| domain.lagrange_from_vec(poly)),
    );

    let vk = match vk {
        Some(vk) => vk,
        None => {
            let permutation_vk = assembly
                .permutation
                .as_ref()
                .expect("permutation must be Some")
                .clone()
                .build_vk(params, &domain, &cs.permutation);

            let fixed_commitments = fixed
                .iter()
                .map(|poly| params.commit_lagrange(poly, Blind::default()).to_affine())
                .collect();

            VerifyingKey::from_parts(
                domain,
                fixed_commitments,
                permutation_vk,
                cs.clone(),
                //                assembly.selectors.clone(),
            )
        }
    };

    let fixed_polys: Vec<_> = fixed
        .iter()
        .map(|poly| vk.domain.lagrange_to_coeff(poly.clone()))
        .collect();

    let permutation_pk = assembly
        .permutation
        .take()
        .expect("permutation must be Some")
        .build_pk(params, &vk.domain, &cs.permutation);

    // Compute l_0(X)
    // TODO: this can be done more efficiently
    let mut l0 = vk.domain.empty_lagrange();
    l0[0] = C::Scalar::ONE;
    let l0 = vk.domain.lagrange_to_coeff(l0);

    // Compute l_blind(X) which evaluates to 1 for each blinding factor row
    // and 0 otherwise over the domain.
    let mut l_blind = vk.domain.empty_lagrange();
    for evaluation in l_blind[..].iter_mut().rev().take(cs.blinding_factors()) {
        *evaluation = C::Scalar::ONE;
    }

    // Compute l_last(X) which evaluates to 1 on the first inactive row (just
    // before the blinding factors) and 0 otherwise over the domain
    let mut l_last = vk.domain.empty_lagrange();
    l_last[params.n() as usize - cs.blinding_factors() - 1] = C::Scalar::ONE;

    // Compute l_active_row(X)
    let one = C::Scalar::ONE;
    let mut l_active_row = vk.domain.empty_lagrange();
    parallelize(&mut l_active_row, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = i + start;
            *value = one - (l_last[idx] + l_blind[idx]);
        }
    });

    let l_last = vk.domain.lagrange_to_coeff(l_last);
    let l_active_row = vk.domain.lagrange_to_coeff(l_active_row);

    // Compute the optimized evaluation data structure
    let ev = Evaluator::new(&vk.cs);

    Ok(ProvingKey {
        vk,
        l0,
        l_last,
        l_active_row,
        fixed_values: fixed,
        fixed_polys,
        permutation: permutation_pk,
        ev,
    })
}
