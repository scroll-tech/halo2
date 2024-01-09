use crate::plonk::{Any, Column};
use crate::poly::Polynomial;
use ff::{FromUniformBytes, PrimeField};
use halo2curves::{serde::SerdeObject, CurveAffine};
use num_bigint::BigUint;
use std::io;

/// This enum specifies how various types are serialized and deserialized.
#[derive(Clone, Copy, Debug)]
pub enum SerdeFormat {
    /// Curve elements are serialized in compressed form.
    /// Field elements are serialized in standard form, with endianness specified by the
    /// `PrimeField` implementation.
    Processed,
    /// Curve elements are serialized in uncompressed form. Field elements are serialized
    /// in their internal Montgomery representation.
    /// When deserializing, checks are performed to ensure curve elements indeed lie on the curve and field elements
    /// are less than modulus.
    RawBytes,
    /// Serialization is the same as `RawBytes`, but no checks are performed.
    RawBytesUnchecked,
}

#[derive(Clone, Debug)]
pub(crate) struct CopyCell {
    pub column: Column<Any>,
    pub row: usize,
}

// Keep this trait for compatibility with IPA serialization
pub(crate) trait CurveRead: CurveAffine {
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut compressed = Self::Repr::default();
        reader.read_exact(compressed.as_mut())?;
        Option::from(Self::from_bytes(&compressed))
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Invalid point encoding in proof"))
    }
}
impl<C: CurveAffine> CurveRead for C {}

pub fn field_to_bn<F: PrimeField>(f: &F) -> BigUint {
    BigUint::from_bytes_le(f.to_repr().as_ref())
}

/// Input a big integer `bn`, compute a field element `f`
/// such that `f == bn % F::MODULUS`.
/// Require:
/// - bn is less than 512 bits.
/// Return:
/// - bn mod F::MODULUS when bn > F::MODULUS
pub fn bn_to_field<F: PrimeField>(bn: &BigUint) -> F
where
    F: FromUniformBytes<64>,
{
    let mut buf = bn.to_bytes_le();
    buf.resize(64, 0u8);

    let mut buf_array = [0u8; 64];
    buf_array.copy_from_slice(buf.as_ref());
    F::from_uniform_bytes(&buf_array)
}

/// Input a base field element `b`, output a scalar field
/// element `s` s.t. `s == b % ScalarField::MODULUS`
pub(crate) fn base_to_scalar<C: CurveAffine>(base: &C::Base) -> C::Scalar
where
    C::Scalar: FromUniformBytes<64>,
{
    let bn = field_to_bn(base);
    // bn_to_field will perform a mod reduction
    bn_to_field(&bn)
}

#[macro_export]
macro_rules! two_dim_vec_to_vec_of_slice {
    ($arc_vec:ident) => {
        unsafe {
            let arc_vec_clone = $arc_vec.clone();
            let ptr = Arc::as_ptr(&arc_vec_clone) as *mut Vec<Vec<_>>;
            let mut_ref = &mut (*ptr);

            mut_ref
                .iter_mut()
                .map(|item| item.as_mut_slice())
                .collect::<Vec<_>>()
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use ff::Field;
    use halo2curves::bn256::{Fq, G1Affine};
    use rand_core::OsRng;
    #[test]
    fn test_conversion() {
        // random numbers
        for _ in 0..100 {
            let b = Fq::random(OsRng);
            let bi = field_to_bn(&b);
            let b_rec = bn_to_field(&bi);
            assert_eq!(b, b_rec);

            let s = base_to_scalar::<G1Affine>(&b);
            let si = field_to_bn(&s);
            // TODO: fixme -- this test has a small probability to fail
            // because |base field| > |scalar field|
            assert_eq!(si, bi);
        }
    }
}

pub trait SerdeCurveAffine: CurveAffine + SerdeObject {
    /// Reads an element from the buffer and parses it according to the `format`:
    /// - `Processed`: Reads a compressed curve element and decompress it
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in Montgomery form.
    /// Checks that field elements are less than modulus, and then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with coordinates in Montgomery form;
    /// does not perform any checks
    fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        match format {
            SerdeFormat::Processed => <Self as CurveRead>::read(reader),
            SerdeFormat::RawBytes => <Self as SerdeObject>::read_raw(reader),
            SerdeFormat::RawBytesUnchecked => Ok(<Self as SerdeObject>::read_raw_unchecked(reader)),
        }
    }
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element
    /// - Otherwise: Writes an uncompressed curve element with coordinates in Montgomery form
    fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        match format {
            SerdeFormat::Processed => writer.write_all(self.to_bytes().as_ref()),
            _ => self.write_raw(writer),
        }
    }
}
impl<C: CurveAffine + SerdeObject> SerdeCurveAffine for C {}

pub trait SerdePrimeField: PrimeField + SerdeObject {
    /// Reads a field element as bytes from the buffer according to the `format`:
    /// - `Processed`: Reads a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation, and checks that the element is less than the modulus.
    /// - `RawBytes`: Reads a field element from raw bytes in its internal Montgomery representations,
    /// and checks that the element is less than the modulus.
    /// - `RawBytesUnchecked`: Reads a field element in Montgomery form and performs no checks.
    fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        match format {
            SerdeFormat::Processed => {
                let mut compressed = Self::Repr::default();
                reader.read_exact(compressed.as_mut())?;
                Option::from(Self::from_repr(compressed)).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::Other, "Invalid prime field point encoding")
                })
            }
            SerdeFormat::RawBytes => <Self as SerdeObject>::read_raw(reader),
            SerdeFormat::RawBytesUnchecked => Ok(<Self as SerdeObject>::read_raw_unchecked(reader)),
        }
    }

    /// Writes a field element as bytes to the buffer according to the `format`:
    /// - `Processed`: Writes a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation.
    /// - Otherwise: Writes a field element into raw bytes in its internal Montgomery representation,
    /// WITHOUT performing the expensive Montgomery reduction.
    fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        match format {
            SerdeFormat::Processed => writer.write_all(self.to_repr().as_ref()),
            _ => self.write_raw(writer),
        }
    }
}
impl<F: PrimeField + SerdeObject> SerdePrimeField for F {}

/// Convert a slice of `bool` into a `u8`.
///
/// Panics if the slice has length greater than 8.
#[allow(unused)]
pub fn pack(bits: &[bool]) -> u8 {
    let mut value = 0u8;
    assert!(bits.len() <= 8);
    for (bit_index, bit) in bits.iter().enumerate() {
        value |= (*bit as u8) << bit_index;
    }
    value
}

/// Writes the first `bits.len()` bits of a `u8` into `bits`.
#[allow(unused)]
pub fn unpack(byte: u8, bits: &mut [bool]) {
    for (bit_index, bit) in bits.iter_mut().enumerate() {
        *bit = (byte >> bit_index) & 1 == 1;
    }
}

/// Reads a vector of polynomials from buffer
pub(crate) fn read_polynomial_vec<R: io::Read, F: SerdePrimeField, B>(
    reader: &mut R,
    format: SerdeFormat,
) -> io::Result<Vec<Polynomial<F, B>>> {
    let mut len = [0u8; 4];
    reader.read_exact(&mut len)?;
    let len = u32::from_be_bytes(len);

    (0..len)
        .map(|_| Polynomial::<F, B>::read(reader, format))
        .collect::<io::Result<Vec<_>>>()
}

/// Writes a slice of polynomials to buffer
pub(crate) fn write_polynomial_slice<W: io::Write, F: SerdePrimeField, B>(
    slice: &[Polynomial<F, B>],
    writer: &mut W,
    format: SerdeFormat,
) -> io::Result<()> {
    writer.write_all(&(slice.len() as u32).to_be_bytes())?;
    for poly in slice.iter() {
        poly.write(writer, format)?;
    }
    Ok(())
}

/// Gets the total number of bytes of a slice of polynomials, assuming all polynomials are the same length
pub(crate) fn polynomial_slice_byte_length<F: PrimeField, B>(slice: &[Polynomial<F, B>]) -> usize {
    let field_len = F::default().to_repr().as_ref().len();
    4 + slice.len() * (4 + field_len * slice.get(0).map(|poly| poly.len()).unwrap_or(0))
}
