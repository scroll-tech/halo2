use std::cmp;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::time::Instant;

use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use ff::Field;

use ark_std::{end_timer, start_timer};

use crate::{
    circuit::{
        layouter::{RegionColumn, RegionLayouter, RegionShape, SyncDeps, TableLayouter},
        table_layouter::{compute_table_lengths, SimpleTableLayouter},
        Cell, Layouter, Region, RegionIndex, RegionStart, Table, Value,
    },
    plonk::{
        Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, Error, Fixed, FloorPlanner,
        Instance, Selector, TableColumn,
    },
};

/// A simple [`FloorPlanner`] that performs minimal optimizations.
///
/// This floor planner is suitable for debugging circuits. It aims to reflect the circuit
/// "business logic" in the circuit layout as closely as possible. It uses a single-pass
/// layouter that does not reorder regions for optimal packing.
#[derive(Debug)]
pub struct SimpleFloorPlanner;

impl FloorPlanner for SimpleFloorPlanner {
    fn synthesize<F: Field, CS: Assignment<F> + SyncDeps, C: Circuit<F>>(
        cs: &mut CS,
        circuit: &C,
        config: C::Config,
        constants: Vec<Column<Fixed>>,
    ) -> Result<(), Error> {
        let timer = start_timer!(|| ("SimpleFloorPlanner synthesize").to_string());
        let layouter = SingleChipLayouter::new(cs, constants)?;
        let result = circuit.synthesize(config, layouter);
        end_timer!(timer);
        result
    }
}

/// A [`Layouter`] for a single-chip circuit.
pub struct SingleChipLayouter<'a, F: Field, CS: Assignment<F> + 'a> {
    cs: &'a mut CS,
    constants: Vec<Column<Fixed>>,
    /// Stores the starting row for each region.
    regions: Vec<RegionStart>,
    /// Stores the first empty row for each column.
    columns: HashMap<RegionColumn, usize>,
    /// Stores the table fixed columns.
    table_columns: Vec<TableColumn>,
    _marker: PhantomData<F>,
}

impl<'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for SingleChipLayouter<'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleChipLayouter")
            .field("regions", &self.regions)
            .field("columns", &self.columns)
            .finish()
    }
}

impl<'a, F: Field, CS: Assignment<F> + 'a> SingleChipLayouter<'a, F, CS> {
    /// Creates a new single-chip layouter.
    pub fn new(cs: &'a mut CS, constants: Vec<Column<Fixed>>) -> Result<Self, Error> {
        let ret = SingleChipLayouter {
            cs,
            constants,
            regions: vec![],
            columns: HashMap::default(),
            table_columns: vec![],
            _marker: PhantomData,
        };
        Ok(ret)
    }

    #[allow(dead_code)]
    fn fork(&self, sub_cs: Vec<&'a mut CS>) -> Result<Vec<Self>, Error> {
        Ok(sub_cs
            .into_iter()
            .map(|sub_cs| Self {
                cs: sub_cs,
                constants: self.constants.clone(),
                regions: self.regions.clone(),
                columns: self.columns.clone(),
                table_columns: self.table_columns.clone(),
                _marker: Default::default(),
            })
            .collect::<Vec<_>>())
    }
}

impl<'a, F: Field, CS: Assignment<F> + 'a + SyncDeps> Layouter<F>
    for SingleChipLayouter<'a, F, CS>
{
    type Root = Self;

    fn assign_region<A, AR, N, NR>(&mut self, name: N, mut assignment: A) -> Result<AR, Error>
    where
        A: FnMut(Region<'_, F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        let region_name: String = name().into();
        let timer = start_timer!(|| format!("assign region: {}", region_name));
        let region_index = self.regions.len();

        // Get shape of the region.
        let mut shape = RegionShape::new(region_index.into());
        {
            let timer_1st = start_timer!(|| format!("assign region 1st pass: {}", region_name));
            let region: &mut dyn RegionLayouter<F> = &mut shape;
            assignment(region.into())?;
            end_timer!(timer_1st);
        }
        let row_count = shape.row_count();
        let log_region_info = row_count >= 40;
        if log_region_info {
            log::debug!(
                "region \"{}\" row_count: {}",
                region_name,
                shape.row_count()
            );
        }

        // Lay out this region. We implement the simplest approach here: position the
        // region starting at the earliest row for which none of the columns are in use.
        let mut region_start = 0;
        for column in &shape.columns {
            let column_start = self.columns.get(column).cloned().unwrap_or(0);
            if column_start != 0 && log_region_info {
                log::trace!(
                    "columns {:?} reused between multi regions. Start: {}. Region: \"{}\"",
                    column,
                    column_start,
                    region_name
                );
            }
            region_start = cmp::max(region_start, column_start);
        }
        if log_region_info {
            log::debug!(
                "region \"{}\", idx {} start {}",
                region_name,
                self.regions.len(),
                region_start
            );
        }
        self.regions.push(region_start.into());

        // Update column usage information.
        for column in shape.columns {
            self.columns.insert(column, region_start + shape.row_count);
        }

        // Assign region cells.
        self.cs.enter_region(name);
        let mut region = SingleChipLayouterRegion::new(self, region_index.into());
        let result = {
            let timer_2nd = start_timer!(|| format!("assign region 2nd pass: {}", region_name));
            let region: &mut dyn RegionLayouter<F> = &mut region;
            let result = assignment(region.into());
            end_timer!(timer_2nd);
            result
        }?;
        let constants_to_assign = region.constants;
        self.cs.exit_region();

        // Assign constants. For the simple floor planner, we assign constants in order in
        // the first `constants` column.
        if self.constants.is_empty() {
            if !constants_to_assign.is_empty() {
                return Err(Error::NotEnoughColumnsForConstants);
            }
        } else {
            let constants_column = self.constants[0];
            let next_constant_row = self
                .columns
                .entry(Column::<Any>::from(constants_column).into())
                .or_default();
            for (constant, advice) in constants_to_assign {
                self.cs.assign_fixed(
                    || format!("Constant({:?})", constant.evaluate()),
                    constants_column,
                    *next_constant_row,
                    || Value::known(constant),
                )?;
                self.cs.copy(
                    constants_column.into(),
                    *next_constant_row,
                    advice.column,
                    *self.regions[*advice.region_index] + advice.row_offset,
                )?;
                *next_constant_row += 1;
            }
        }

        end_timer!(timer);
        Ok(result)
    }

    #[cfg(feature = "parallel_syn")]
    fn assign_regions<A, AR, N, NR>(
        &mut self,
        name: N,
        mut assignments: Vec<A>,
    ) -> Result<Vec<AR>, Error>
    where
        A: FnMut(Region<'_, F>) -> Result<AR, Error> + Send,
        AR: Send,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        let region_index = self.regions.len();
        let region_name: String = name().into();
        // Get region shapes sequentially
        let mut ranges = vec![];
        for (i, assignment) in assignments.iter_mut().enumerate() {
            // Get shape of the ith sub-region.
            let mut shape = RegionShape::new((region_index + i).into());
            let region: &mut dyn RegionLayouter<F> = &mut shape;
            assignment(region.into())?;

            let mut region_start = 0;
            for column in &shape.columns {
                let column_start = self.columns.get(column).cloned().unwrap_or(0);
                region_start = cmp::max(region_start, column_start);
            }
            log::debug!(
                "{}_{} start: {}, end: {}",
                region_name,
                i,
                region_start,
                region_start + shape.row_count()
            );
            self.regions.push(region_start.into());
            ranges.push(region_start..(region_start + shape.row_count()));

            // Update column usage information.
            for column in shape.columns.iter() {
                self.columns
                    .insert(*column, region_start + shape.row_count());
            }
        }

        // Do actual synthesis of sub-regions in parallel
        let cs_fork_time = Instant::now();
        let mut sub_cs = self.cs.fork(&ranges)?;
        log::debug!(
            "CS forked into {} subCS took {:?}",
            sub_cs.len(),
            cs_fork_time.elapsed()
        );
        let ref_sub_cs = sub_cs.iter_mut().collect();
        let sub_layouters = self.fork(ref_sub_cs)?;
        let regions_2nd_pass = Instant::now();
        let ret = assignments
            .into_par_iter()
            .zip(sub_layouters.into_par_iter())
            .enumerate()
            .map(|(i, (mut assignment, mut sub_layouter))| {
                let region_name = format!("{}_{}", region_name, i);
                let sub_region_2nd_pass = Instant::now();
                sub_layouter.cs.enter_region(|| region_name.clone());
                let mut region =
                    SingleChipLayouterRegion::new(&mut sub_layouter, (region_index + i).into());
                let region_ref: &mut dyn RegionLayouter<F> = &mut region;
                let result = assignment(region_ref.into());
                let constant = region.constants.clone();
                sub_layouter.cs.exit_region();
                log::debug!(
                    "region {} 2nd pass synthesis took {:?}",
                    region_name,
                    sub_region_2nd_pass.elapsed()
                );
                (result, constant)
            })
            .collect::<Vec<_>>();
        let cs_merge_time = Instant::now();
        let num_sub_cs = sub_cs.len();
        self.cs.merge(sub_cs)?;
        log::debug!(
            "Merge {} subCS back took {:?}",
            num_sub_cs,
            cs_merge_time.elapsed()
        );
        log::debug!(
            "{} sub_regions of {} 2nd pass synthesis took {:?}",
            ranges.len(),
            region_name,
            regions_2nd_pass.elapsed()
        );
        let (results, constants): (Vec<_>, Vec<_>) = ret.into_iter().unzip();

        // Check if there are errors in sub-region synthesis
        let results = results.into_iter().collect::<Result<Vec<_>, Error>>()?;

        // Merge all constants from sub-regions together
        let constants_to_assign = constants
            .into_iter()
            .flat_map(|constant_to_assign| constant_to_assign.into_iter())
            .collect::<Vec<_>>();

        // Assign constants. For the simple floor planner, we assign constants in order in
        // the first `constants` column.
        if self.constants.is_empty() {
            if !constants_to_assign.is_empty() {
                return Err(Error::NotEnoughColumnsForConstants);
            }
        } else {
            let constants_column = self.constants[0];
            let next_constant_row = self
                .columns
                .entry(Column::<Any>::from(constants_column).into())
                .or_default();
            for (constant, advice) in constants_to_assign {
                self.cs.assign_fixed(
                    || format!("Constant({:?})", constant.evaluate()),
                    constants_column,
                    *next_constant_row,
                    || Value::known(constant),
                )?;
                self.cs.copy(
                    constants_column.into(),
                    *next_constant_row,
                    advice.column,
                    *self.regions[*advice.region_index] + advice.row_offset,
                )?;
                *next_constant_row += 1;
            }
        }

        Ok(results)
    }

    fn assign_table<A, N, NR>(&mut self, name: N, mut assignment: A) -> Result<(), Error>
    where
        A: FnMut(Table<'_, F>) -> Result<(), Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        // Maintenance hazard: there is near-duplicate code in `v1::AssignmentPass::assign_table`.
        // Assign table cells.
        self.cs.enter_region(name);
        let mut table = SimpleTableLayouter::new(self.cs, &self.table_columns);
        {
            let table: &mut dyn TableLayouter<F> = &mut table;
            assignment(table.into())
        }?;
        let default_and_assigned = table.default_and_assigned;
        self.cs.exit_region();

        // Check that all table columns have the same length `first_unused`,
        // and all cells up to that length are assigned.
        let first_unused = compute_table_lengths(&default_and_assigned)?;

        // Record these columns so that we can prevent them from being used again.
        for column in default_and_assigned.keys() {
            self.table_columns.push(*column);
        }

        for (col, (default_val, _)) in default_and_assigned {
            // default_val must be Some because we must have assigned
            // at least one cell in each column, and in that case we checked
            // that all cells up to first_unused were assigned.
            self.cs
                .fill_from_row(col.inner(), first_unused, default_val.unwrap())?;
        }

        Ok(())
    }

    fn constrain_instance(
        &mut self,
        cell: Cell,
        instance: Column<Instance>,
        row: usize,
    ) -> Result<(), Error> {
        self.cs.copy(
            cell.column,
            *self.regions[*cell.region_index] + cell.row_offset,
            instance.into(),
            row,
        )
    }

    fn get_challenge(&self, challenge: Challenge) -> Value<F> {
        self.cs.get_challenge(challenge)
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.cs.push_namespace(name_fn)
    }

    fn pop_namespace(&mut self, gadget_name: Option<String>) {
        self.cs.pop_namespace(gadget_name)
    }
}

struct SingleChipLayouterRegion<'r, 'a, F: Field, CS: Assignment<F> + 'a> {
    layouter: &'r mut SingleChipLayouter<'a, F, CS>,
    region_index: RegionIndex,
    /// Stores the constants to be assigned, and the cells to which they are copied.
    constants: Vec<(Assigned<F>, Cell)>,
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug
    for SingleChipLayouterRegion<'r, 'a, F, CS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleChipLayouterRegion")
            .field("layouter", &self.layouter)
            .field("region_index", &self.region_index)
            .finish()
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> SingleChipLayouterRegion<'r, 'a, F, CS> {
    fn new(layouter: &'r mut SingleChipLayouter<'a, F, CS>, region_index: RegionIndex) -> Self {
        SingleChipLayouterRegion {
            layouter,
            region_index,
            constants: vec![],
        }
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a + SyncDeps> RegionLayouter<F>
    for SingleChipLayouterRegion<'r, 'a, F, CS>
{
    fn enable_selector<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        selector: &Selector,
        offset: usize,
    ) -> Result<(), Error> {
        self.layouter.cs.enable_selector(
            annotation,
            selector,
            *self.layouter.regions[*self.region_index] + offset,
        )
    }

    fn name_column<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Any>,
    ) {
        self.layouter.cs.annotate_column(annotation, column);
    }

    fn query_advice(&self, column: Column<Advice>, offset: usize) -> Result<F, Error> {
        self.layouter
            .cs
            .query_advice(column, *self.layouter.regions[*self.region_index] + offset)
    }

    fn query_fixed(&self, column: Column<Fixed>, offset: usize) -> Result<F, Error> {
        self.layouter
            .cs
            .query_fixed(column, *self.layouter.regions[*self.region_index] + offset)
    }

    fn assign_advice<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Value<Assigned<F>> + 'v),
    ) -> Result<Cell, Error> {
        self.layouter.cs.assign_advice(
            annotation,
            column,
            *self.layouter.regions[*self.region_index] + offset,
            to,
        )?;

        Ok(Cell {
            region_index: self.region_index,
            row_offset: offset,
            column: column.into(),
        })
    }

    fn assign_advice_from_constant<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        constant: Assigned<F>,
    ) -> Result<Cell, Error> {
        let advice =
            self.assign_advice(annotation, column, offset, &mut || Value::known(constant))?;
        self.constrain_constant(advice, constant)?;

        Ok(advice)
    }

    fn assign_advice_from_instance<'v>(
        &mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        instance: Column<Instance>,
        row: usize,
        advice: Column<Advice>,
        offset: usize,
    ) -> Result<(Cell, Value<F>), Error> {
        let value = self.layouter.cs.query_instance(instance, row)?;

        let cell = self.assign_advice(annotation, advice, offset, &mut || value.to_field())?;

        self.layouter.cs.copy(
            cell.column,
            *self.layouter.regions[*cell.region_index] + cell.row_offset,
            instance.into(),
            row,
        )?;

        Ok((cell, value))
    }

    fn instance_value(
        &mut self,
        instance: Column<Instance>,
        row: usize,
    ) -> Result<Value<F>, Error> {
        self.layouter.cs.query_instance(instance, row)
    }

    fn assign_fixed<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Fixed>,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Value<Assigned<F>> + 'v),
    ) -> Result<Cell, Error> {
        self.layouter.cs.assign_fixed(
            annotation,
            column,
            *self.layouter.regions[*self.region_index] + offset,
            to,
        )?;

        Ok(Cell {
            region_index: self.region_index,
            row_offset: offset,
            column: column.into(),
        })
    }

    fn constrain_constant(&mut self, cell: Cell, constant: Assigned<F>) -> Result<(), Error> {
        self.constants.push((constant, cell));
        Ok(())
    }

    fn constrain_equal(&mut self, left: Cell, right: Cell) -> Result<(), Error> {
        self.layouter.cs.copy(
            left.column,
            *self.layouter.regions[*left.region_index] + left.row_offset,
            right.column,
            *self.layouter.regions[*right.region_index] + right.row_offset,
        )?;

        Ok(())
    }

    fn global_offset(&self, row_offset: usize) -> usize {
        *self.layouter.regions[*self.region_index] + row_offset
    }
}

#[cfg(test)]
mod tests {
    use halo2curves::pasta::vesta;

    use super::SimpleFloorPlanner;
    use crate::{
        dev::MockProver,
        plonk::{Advice, Circuit, Column, Error},
    };

    #[test]
    fn not_enough_columns_for_constants() {
        struct MyCircuit {}

        impl Circuit<vesta::Scalar> for MyCircuit {
            type Config = Column<Advice>;
            type FloorPlanner = SimpleFloorPlanner;
            #[cfg(feature = "circuit-params")]
            type Params = ();

            fn without_witnesses(&self) -> Self {
                MyCircuit {}
            }

            fn configure(meta: &mut crate::plonk::ConstraintSystem<vesta::Scalar>) -> Self::Config {
                meta.advice_column()
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl crate::circuit::Layouter<vesta::Scalar>,
            ) -> Result<(), crate::plonk::Error> {
                layouter.assign_region(
                    || "assign constant",
                    |mut region| {
                        region.assign_advice_from_constant(
                            || "one",
                            config,
                            0,
                            vesta::Scalar::one(),
                        )
                    },
                )?;

                Ok(())
            }
        }

        let circuit = MyCircuit {};
        assert!(matches!(
            MockProver::run(3, &circuit, vec![]).unwrap_err(),
            Error::NotEnoughColumnsForConstants,
        ));
    }
}
