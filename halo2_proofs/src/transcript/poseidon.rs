use super::{Challenge255, EncodedChallenge, Transcript, TranscriptRead, TranscriptWrite};
use crate::helpers::base_to_scalar;
use ff::Field;
use group::ff::PrimeField;
use halo2curves::{Coordinates, CurveAffine, FieldExt};
use num_bigint::BigUint;
use poseidon::Poseidon;
use std::convert::TryInto;
use std::io::{self, Read, Write};
use std::marker::PhantomData;

const POSEIDON_RATE: usize = 8usize;
const POSEIDON_T: usize = POSEIDON_RATE + 1usize;

/// TODO
#[derive(Debug, Clone)]
pub struct PoseidonRead<R: Read, C: CurveAffine, E: EncodedChallenge<C>> {
    state: Poseidon<C::ScalarExt, POSEIDON_T, POSEIDON_RATE>,
    reader: R,
    _marker: PhantomData<(C, E)>,
}

/// TODO
impl<R: Read, C: CurveAffine, E: EncodedChallenge<C>> PoseidonRead<R, C, E> {
    /// Initialize a transcript given an input buffer.
    pub fn init(reader: R) -> Self {
        PoseidonRead {
            state: Poseidon::new(8usize, 63usize),
            reader,
            _marker: PhantomData,
        }
    }
}

impl<R: Read, C: CurveAffine> TranscriptRead<C, Challenge255<C>>
    for PoseidonRead<R, C, Challenge255<C>>
{
    fn read_point(&mut self) -> io::Result<C> {
        let mut compressed = C::Repr::default();
        self.reader.read_exact(compressed.as_mut())?;
        let point: C = Option::from(C::from_bytes(&compressed)).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "invalid point encoding in proof for poseidon",
            )
        })?;
        self.common_point(point)?;

        Ok(point)
    }

    fn read_scalar(&mut self) -> io::Result<C::Scalar> {
        let mut data = <C::Scalar as PrimeField>::Repr::default();
        self.reader.read_exact(data.as_mut())?;
        let scalar: C::Scalar = Option::from(C::Scalar::from_repr(data)).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "invalid field element encoding in proof for poseidon",
            )
        })?;
        self.common_scalar(scalar)?;

        Ok(scalar)
    }
}

impl<R: Read, C: CurveAffine> Transcript<C, Challenge255<C>>
    for PoseidonRead<R, C, Challenge255<C>>
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        //self.state.update(&[PREFIX_SQUEEZE]);
        let scalar = self.state.squeeze();
        let mut scalar_bytes = scalar.to_repr().as_ref().to_vec();
        scalar_bytes.resize(64, 0u8);
        Challenge255::<C>::new(&scalar_bytes.try_into().unwrap())
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        //self.state.update(&[PREFIX_POINT]);
        let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot write points at infinity to the transcript",
            )
        })?;
        let x = coords.x();
        let y = coords.y();
        self.state
            .update(&[base_to_scalar::<C>(x), base_to_scalar::<C>(y)]);

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        //self.state.update(&[BLAKE2B_PREFIX_SCALAR]);
        self.state.update(&[scalar]);

        Ok(())
    }
}
/// TODO
#[derive(Debug, Clone)]
pub struct PoseidonWrite<W: Write, C: CurveAffine, E: EncodedChallenge<C>> {
    state: Poseidon<C::ScalarExt, POSEIDON_T, POSEIDON_RATE>,
    writer: W,
    _marker: PhantomData<(C, E)>,
}

impl<W: Write, C: CurveAffine, E: EncodedChallenge<C>> PoseidonWrite<W, C, E> {
    /// Initialize a transcript given an output buffer.
    pub fn init(writer: W) -> Self {
        PoseidonWrite {
            state: Poseidon::new(8usize, 63usize),
            writer,
            _marker: PhantomData,
        }
    }

    /// Conclude the interaction and return the output buffer (writer).
    pub fn finalize(self) -> W {
        // TODO: handle outstanding scalars? see issue #138
        self.writer
    }
}

impl<W: Write, C: CurveAffine> TranscriptWrite<C, Challenge255<C>>
    for PoseidonWrite<W, C, Challenge255<C>>
{
    fn write_point(&mut self, point: C) -> io::Result<()> {
        self.common_point(point)?;
        let compressed = point.to_bytes();
        self.writer.write_all(compressed.as_ref())
    }
    fn write_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.common_scalar(scalar)?;
        let data = scalar.to_repr();
        self.writer.write_all(data.as_ref())
    }
}

impl<W: Write, C: CurveAffine> Transcript<C, Challenge255<C>>
    for PoseidonWrite<W, C, Challenge255<C>>
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        //self.state.update(&[PREFIX_SQUEEZE]);
        let scalar = self.state.squeeze();
        let mut scalar_bytes = scalar.to_repr().as_ref().to_vec();
        scalar_bytes.resize(64, 0u8);
        Challenge255::<C>::new(&scalar_bytes.try_into().unwrap())
    }

    fn common_point(&mut self, point: C) -> io::Result<()> {
        //self.state.update(&[PREFIX_POINT]);
        let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "cannot write points at infinity to the transcript",
            )
        })?;
        let x = coords.x();
        let y = coords.y();
        self.state
            .update(&[base_to_scalar::<C>(x), base_to_scalar::<C>(y)]);

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        //self.state.update(&[BLAKE2B_PREFIX_SCALAR]);
        self.state.update(&[scalar]);

        Ok(())
    }
}
