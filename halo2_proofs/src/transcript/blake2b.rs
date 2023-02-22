use super::{
    Challenge255, EncodedChallenge, Transcript, TranscriptRead, TranscriptReadBuffer,
    TranscriptWrite, TranscriptWriterBuffer,
};
use blake2b_simd::{Params as Blake2bParams, State as Blake2bState};
use ff::Field;
use group::ff::PrimeField;
use halo2curves::{Coordinates, CurveAffine, FieldExt};
use num_bigint::BigUint;
use std::convert::TryInto;
use std::io::{self, Read, Write};
use std::marker::PhantomData;

/// Prefix to a prover's message soliciting a challenge
const BLAKE2B_PREFIX_CHALLENGE: u8 = 0;

/// Prefix to a prover's message containing a curve point
const BLAKE2B_PREFIX_POINT: u8 = 1;

/// Prefix to a prover's message containing a scalar
const BLAKE2B_PREFIX_SCALAR: u8 = 2;

// ----------------------Blake2bRead

/// We will replace BLAKE2b with an algebraic hash function in a later version.
#[derive(Debug, Clone)]
pub struct Blake2bRead<R: Read, C: CurveAffine, E: EncodedChallenge<C>> {
    state: Blake2bState,
    reader: R,
    _marker: PhantomData<(C, E)>,
}

impl<R: Read, C: CurveAffine> TranscriptReadBuffer<R, C, Challenge255<C>>
    for Blake2bRead<R, C, Challenge255<C>>
{
    /// Initialize a transcript given an input buffer.
    fn init(reader: R) -> Self {
        Blake2bRead {
            state: Blake2bParams::new()
                .hash_length(64)
                .personal(b"Halo2-Transcript")
                .to_state(),
            reader,
            _marker: PhantomData,
        }
    }
}

impl<R: Read, C: CurveAffine> TranscriptRead<C, Challenge255<C>>
    for Blake2bRead<R, C, Challenge255<C>>
{
    fn read_point(&mut self) -> io::Result<C> {
        let mut compressed = C::Repr::default();
        self.reader.read_exact(compressed.as_mut())?;
        let point: C = match Option::from(C::from_bytes(&compressed)) {
            Some(p) => p,
            // TODO: check that this is actually safe to push an
            // identity point to the transcript
            None => C::identity(),
        };
        self.common_point(point)?;

        Ok(point)
    }

    fn read_scalar(&mut self) -> io::Result<C::Scalar> {
        let mut data = <C::Scalar as PrimeField>::Repr::default();
        self.reader.read_exact(data.as_mut())?;
        let scalar = match Option::from(C::Scalar::from_repr(data)) {
            Some(p) => p,
            // TODO: check that this is actually safe to push an
            // identity point to the transcript
            None => C::Scalar::zero(),
        };
        self.common_scalar(scalar)?;

        Ok(scalar)
    }
}

impl<R: Read, C: CurveAffine> Transcript<C, Challenge255<C>>
    for Blake2bRead<R, C, Challenge255<C>>
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        self.state.update(&[BLAKE2B_PREFIX_CHALLENGE]);
        let hasher = self.state.clone();
        let result: [u8; 64] = hasher.finalize().as_bytes().try_into().unwrap();
        Challenge255::<C>::new(&result)
    }

    // This function is slightly modified from PSE's version.
    // In PSE's version, an error is returned if the input point is infinity.
    // Here we want to be able to absorb infinity point because of the
    // randomness we used in the polynomial commitment is 0.
    fn common_point(&mut self, point: C) -> io::Result<()> {
        self.state.update(&[BLAKE2B_PREFIX_POINT]);

        let tmp: Option<Coordinates<_>> = Option::from(point.coordinates());
        match tmp {
            Some(coords) => {
                self.state.update(coords.x().to_repr().as_ref());
                self.state.update(coords.y().to_repr().as_ref());
            }
            None => {
                // Infinity point
                self.state.update(C::Base::zero().to_repr().as_ref());
                self.state.update(C::Base::from(5).to_repr().as_ref());
            }
        }
        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.state.update(&[BLAKE2B_PREFIX_SCALAR]);
        self.state.update(scalar.to_repr().as_ref());

        Ok(())
    }
}

/// We will replace BLAKE2b with an algebraic hash function in a later version.
#[derive(Debug, Clone)]
pub struct Blake2bWrite<W: Write, C: CurveAffine, E: EncodedChallenge<C>> {
    state: Blake2bState,
    writer: W,
    _marker: PhantomData<(C, E)>,
}

impl<W: Write, C: CurveAffine> TranscriptWriterBuffer<W, C, Challenge255<C>>
    for Blake2bWrite<W, C, Challenge255<C>>
{
    fn init(writer: W) -> Self {
        Blake2bWrite {
            state: Blake2bParams::new()
                .hash_length(64)
                .personal(b"Halo2-Transcript")
                .to_state(),
            writer,
            _marker: PhantomData,
        }
    }

    fn finalize(self) -> W {
        // TODO: handle outstanding scalars? see issue #138
        self.writer
    }
}

impl<W: Write, C: CurveAffine> TranscriptWrite<C, Challenge255<C>>
    for Blake2bWrite<W, C, Challenge255<C>>
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
    for Blake2bWrite<W, C, Challenge255<C>>
{
    fn squeeze_challenge(&mut self) -> Challenge255<C> {
        self.state.update(&[BLAKE2B_PREFIX_CHALLENGE]);
        let hasher = self.state.clone();
        let result: [u8; 64] = hasher.finalize().as_bytes().try_into().unwrap();
        Challenge255::<C>::new(&result)
    }

    // This function is slightly modified from PSE's version.
    // In PSE's version, an error is returned if the input point is infinity.
    // Here we want to be able to absorb infinity point because of the
    // randomness we used in the polynomial commitment is 0.
    fn common_point(&mut self, point: C) -> io::Result<()> {
        self.state.update(&[BLAKE2B_PREFIX_POINT]);
        let tmp: Option<Coordinates<_>> = Option::from(point.coordinates());
        match tmp {
            Some(coords) => {
                self.state.update(coords.x().to_repr().as_ref());
                self.state.update(coords.y().to_repr().as_ref());
            }
            None => {
                // Infinity point
                self.state.update(C::Base::zero().to_repr().as_ref());
                self.state.update(C::Base::from(5).to_repr().as_ref());
            }
        }

        Ok(())
    }

    fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        self.state.update(&[BLAKE2B_PREFIX_SCALAR]);
        self.state.update(scalar.to_repr().as_ref());

        Ok(())
    }
}
