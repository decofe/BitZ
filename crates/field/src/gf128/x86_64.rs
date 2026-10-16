//! PCLMULQDQ kernels for `GF(2^128)`.
//!
//! `pclmulqdq` does a 64x64 carryless multiply in one instruction, replacing
//! the 64-step loop in [`super::portable::clmul64`]. The schoolbook product and
//! the reduction are the portable pipeline with that substitution.
//!
//! The module compiles only with `pclmulqdq` enabled crate-wide (for example
//! `-C target-cpu=native` or `-C target-feature=+pclmulqdq`), so every
//! intrinsic call below is sound on that basis.

use core::arch::x86_64::{
    _mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_set_epi64x, _mm_unpackhi_epi64,
};

use super::portable::reduce;

pub use super::portable::{Wide, wide_add, wide_add_low, wide_of, wide_reduce, wide_zero};

/// Carryless product of two 64-bit polynomials, as `(low, high)` halves.
#[inline(always)]
fn clmul64(a: u64, b: u64) -> (u64, u64) {
    unsafe {
        let product =
            _mm_clmulepi64_si128::<0x00>(_mm_set_epi64x(0, a as i64), _mm_set_epi64x(0, b as i64));
        (
            _mm_cvtsi128_si64(product) as u64,
            _mm_cvtsi128_si64(_mm_unpackhi_epi64(product, product)) as u64,
        )
    }
}

/// Schoolbook 128x128 carryless product, four words in ascending significance.
#[inline(always)]
fn clmul128(a: [u64; 2], b: [u64; 2]) -> [u64; 4] {
    let (l0, l1) = clmul64(a[0], b[0]);
    let (h0, h1) = clmul64(a[1], b[1]);
    let (m0, m1) = clmul64(a[0], b[1]);
    let (n0, n1) = clmul64(a[1], b[0]);
    [l0, l1 ^ m0 ^ n0, h0 ^ m1 ^ n1, h1]
}

#[inline]
pub fn mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    reduce(clmul128(a, b))
}

/// In characteristic 2 the cross terms cancel, so squaring needs 2 products.
#[inline]
pub fn square(a: [u64; 2]) -> [u64; 2] {
    let (l0, l1) = clmul64(a[0], a[0]);
    let (h0, h1) = clmul64(a[1], a[1]);
    reduce([l0, l1, h0, h1])
}

#[inline]
pub fn square_n(a: [u64; 2], k: u32) -> [u64; 2] {
    let mut w = a;
    for _ in 0..k {
        w = square(w);
    }
    w
}

#[inline]
pub fn wide_mul(a: [u64; 2], b: [u64; 2]) -> Wide {
    clmul128(a, b)
}
