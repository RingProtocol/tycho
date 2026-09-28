//! Native swap-to-price solver for StableSwap pools. Not part of the vendored `curve-math` tree.
//!
//! Runs Illinois false position on the input amount with the pool's invariant math.

use alloy_primitives::{U256, U512};

use crate::evm::protocol::{
    curve::math::{core, Pool},
    u256_num::u256_to_f64,
};

const PRECISION: U256 = U256::from_limbs([1_000_000_000_000_000_000, 0, 0, 0]);
const FEE_DENOMINATOR: U256 = U256::from_limbs([10_000_000_000, 0, 0, 0]);
const MAX_ITERATIONS: usize = 32;
/// Maximum halvings of the upper search bound when the pool math fails at the full balance.
const MAX_BOUND_HALVINGS: usize = 4;

/// Error returned by [`swap_to_price`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SwapToPriceError {
    /// CryptoSwap variants have no native solver.
    #[error("no native swap-to-price solver for this pool variant")]
    UnsupportedVariant,
    /// Selling coin `i` only lowers the price, so no swap reaches a target above spot.
    #[error("target price is above the current spot price")]
    TargetAboveSpot,
    /// Selling the pool's whole balance of coin `i` does not lower the price to the target.
    #[error("target price is below the pool's reachable limit")]
    TargetBelowLimit,
    /// The pool math overflowed or failed, or the search ended outside the tolerance band.
    #[error("pool math failed or the search did not reach the tolerance band")]
    MathFailed,
    /// The coin indices are equal or out of range, or the target fraction has a zero part.
    #[error("invalid swap-to-price input: {0}")]
    InvalidInput(String),
}

type GetDFn = fn(&[U256], U256) -> Option<U256>;
type GetYFn = fn(usize, usize, U256, &[U256], U256, U256) -> Option<U256>;
type DynamicFeeFn = fn(U256, U256, U256, U256) -> U256;

/// The bracket endpoint that the last false-position step moved.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LastMoved {
    Neither,
    Lo,
    Hi,
}

/// The `get_d` and `get_y` functions and the quote rules of one StableSwap variant.
struct VariantMath {
    get_d: GetDFn,
    /// `get_y(i, j, x_new, xp, d, amp)` returns the new normalized balance of coin `j`.
    get_y: GetYFn,
    a_precision: U256,
    /// Whether the variant subtracts 1 wei from `xp[j] - y_new` (V1/V2/STETH/NG/Meta).
    minus_one_offset: bool,
    /// Charges the fee on normalized output (V2 and later) instead of denormalized (V0/V1/ALend).
    fee_before_denorm: bool,
    /// `(xp_i, xp_j, fee, offpeg_fee_multiplier) -> fee` for NG/ALend.
    dynamic_fee: Option<DynamicFeeFn>,
}

const V0_MATH: VariantMath = VariantMath {
    get_d: core::stableswap_v0::get_d,
    get_y: core::stableswap_v0::get_y,
    a_precision: core::stableswap_v0::A_PRECISION,
    minus_one_offset: false,
    fee_before_denorm: false,
    dynamic_fee: None,
};

const V1_MATH: VariantMath = VariantMath {
    get_d: core::stableswap_v1::get_d,
    get_y: core::stableswap_v1::get_y,
    a_precision: core::stableswap_v1::A_PRECISION,
    minus_one_offset: true,
    fee_before_denorm: false,
    dynamic_fee: None,
};

const V2_MATH: VariantMath = VariantMath {
    get_d: core::stableswap_v2::get_d,
    get_y: core::stableswap_v2::get_y,
    a_precision: core::stableswap_v2::A_PRECISION,
    minus_one_offset: true,
    fee_before_denorm: true,
    dynamic_fee: None,
};

const STETH_MATH: VariantMath = VariantMath {
    get_d: core::stableswap_steth::get_d,
    get_y: core::stableswap_steth::get_y,
    a_precision: core::stableswap_steth::A_PRECISION,
    minus_one_offset: true,
    fee_before_denorm: true,
    dynamic_fee: None,
};

const ALEND_MATH: VariantMath = VariantMath {
    get_d: core::stableswap_alend::get_d,
    get_y: core::stableswap_alend::get_y,
    a_precision: core::stableswap_alend::A_PRECISION,
    minus_one_offset: false,
    fee_before_denorm: false,
    dynamic_fee: Some(core::stableswap_alend::dynamic_fee),
};

const NG_MATH: VariantMath = VariantMath {
    get_d: core::stableswap_ng::get_d,
    get_y: core::stableswap_ng::get_y,
    a_precision: core::stableswap_ng::A_PRECISION,
    minus_one_offset: true,
    fee_before_denorm: true,
    dynamic_fee: Some(core::stableswap_ng::dynamic_fee),
};

const META_MATH: VariantMath = VariantMath {
    get_d: core::stableswap_meta::get_d,
    get_y: core::stableswap_meta::get_y,
    a_precision: core::stableswap_meta::A_PRECISION,
    minus_one_offset: true,
    fee_before_denorm: true,
    dynamic_fee: None,
};

/// A StableSwap pool in one layout for every variant. `rates` are 1e18-scaled, so ALend's
/// `precision_mul` becomes `precision_mul * PRECISION`.
struct NormalizedStablePool<'p> {
    balances: &'p [U256],
    rates: Vec<U256>,
    amp: U256,
    fee: U256,
    offpeg_fee_multiplier: U256,
    math: VariantMath,
}

/// Returns the amount of coin `i` to sell so that [`Pool::spot_price`]`(i, j)` lands in
/// `[target, target * (1 + tolerance)]`, with the target `target_num / target_den` in native
/// units. A target equal to spot returns zero.
pub fn swap_to_price(
    pool: &Pool,
    i: usize,
    j: usize,
    target_num: U256,
    target_den: U256,
    tolerance: f64,
) -> Result<U256, SwapToPriceError> {
    let Some(normalized) = NormalizedStablePool::from_pool(pool) else {
        return Err(SwapToPriceError::UnsupportedVariant);
    };
    normalized.solve(i, j, target_num, target_den, tolerance)
}

impl<'p> NormalizedStablePool<'p> {
    /// Returns `None` for CryptoSwap variants.
    fn from_pool(pool: &'p Pool) -> Option<Self> {
        let (balances, rates, amp, fee, offpeg_fee_multiplier, math) = match pool {
            Pool::StableSwapV0 { balances, rates, amp, fee } => {
                (balances, rates.clone(), amp, fee, U256::ZERO, V0_MATH)
            }
            Pool::StableSwapV1 { balances, rates, amp, fee } => {
                (balances, rates.clone(), amp, fee, U256::ZERO, V1_MATH)
            }
            Pool::StableSwapV2 { balances, rates, amp, fee } => {
                (balances, rates.clone(), amp, fee, U256::ZERO, V2_MATH)
            }
            Pool::StableSwapSTETH { balances, rates, amp, fee } => {
                (balances, rates.clone(), amp, fee, U256::ZERO, STETH_MATH)
            }
            Pool::StableSwapALend { balances, precision_mul, amp, fee, offpeg_fee_multiplier } => {
                let rates = precision_mul
                    .iter()
                    .map(|p| *p * PRECISION)
                    .collect();
                (balances, rates, amp, fee, *offpeg_fee_multiplier, ALEND_MATH)
            }
            Pool::StableSwapNG { balances, rates, amp, fee, offpeg_fee_multiplier } => {
                (balances, rates.clone(), amp, fee, *offpeg_fee_multiplier, NG_MATH)
            }
            Pool::StableSwapMeta { balances, rates, amp, fee } => {
                (balances, rates.clone(), amp, fee, U256::ZERO, META_MATH)
            }
            Pool::TwoCryptoV1 { .. } |
            Pool::TwoCryptoNG { .. } |
            Pool::TwoCryptoStable { .. } |
            Pool::TriCryptoV1 { .. } |
            Pool::TriCryptoNG { .. } => return None,
        };
        Some(NormalizedStablePool {
            balances,
            rates,
            amp: *amp,
            fee: *fee,
            offpeg_fee_multiplier,
            math,
        })
    }

    fn solve(
        &self,
        i: usize,
        j: usize,
        target_num: U256,
        target_den: U256,
        tolerance: f64,
    ) -> Result<U256, SwapToPriceError> {
        let n = self.balances.len();
        if i >= n || j >= n || i == j {
            return Err(SwapToPriceError::InvalidInput(format!(
                "coin indices {i} and {j} must differ and be below {n}"
            )));
        }
        if target_num.is_zero() || target_den.is_zero() {
            return Err(SwapToPriceError::InvalidInput(format!(
                "target fraction {target_num}/{target_den} has a zero part"
            )));
        }

        let xp = self
            .xp(self.balances)
            .ok_or(SwapToPriceError::MathFailed)?;
        let d = (self.math.get_d)(&xp, self.amp).ok_or(SwapToPriceError::MathFailed)?;
        let spot = self
            .price_fraction(&xp, self.balances, d, i, j)
            .ok_or(SwapToPriceError::MathFailed)?;

        match fraction_cmp(&spot, target_num, target_den) {
            std::cmp::Ordering::Less => return Err(SwapToPriceError::TargetAboveSpot),
            std::cmp::Ordering::Equal => return Ok(U256::ZERO),
            std::cmp::Ordering::Greater => {}
        }

        let (hi, limit_price) = self
            .upper_bound(&xp, d, i, j)
            .ok_or(SwapToPriceError::MathFailed)?;
        match fraction_cmp(&limit_price, target_num, target_den) {
            std::cmp::Ordering::Greater => return Err(SwapToPriceError::TargetBelowLimit),
            std::cmp::Ordering::Equal => return Ok(hi),
            std::cmp::Ordering::Less => {}
        }

        let band = TargetBand {
            num: target_num,
            den: target_den,
            target: fraction_to_f64(&(target_num, target_den))
                .ok_or(SwapToPriceError::MathFailed)?,
            tolerance,
        };
        let spot_f = fraction_to_f64(&spot).ok_or(SwapToPriceError::MathFailed)?;
        let limit_f = fraction_to_f64(&limit_price).ok_or(SwapToPriceError::MathFailed)?;
        let bracket = Bracket {
            lo: U256::ZERO,
            hi,
            g_lo: spot_f - band.aim(),
            g_hi: limit_f - band.aim(),
            last_moved: LastMoved::Neither,
        };
        false_position(|dx| self.post_swap_price(&xp, d, i, j, dx), &band, bracket)
    }

    /// Returns the upper search bound and the post-swap price there. The bound starts at the
    /// balance of coin `i`, the soft limit of `get_limits`, and halves while the math fails.
    fn upper_bound(
        &self,
        xp: &[U256],
        d: U256,
        i: usize,
        j: usize,
    ) -> Option<(U256, (U256, U256))> {
        let mut hi = self.balances[i];
        for _ in 0..MAX_BOUND_HALVINGS {
            if hi.is_zero() {
                return None;
            }
            if let Some((price, _dy)) = self.post_swap_price(xp, d, i, j, hi) {
                return Some((hi, price));
            }
            hi /= U256::from(2);
        }
        None
    }

    /// Normalized balances `xp[k] = balances[k] * rates[k] / PRECISION`.
    fn xp(&self, balances: &[U256]) -> Option<Vec<U256>> {
        balances
            .iter()
            .zip(self.rates.iter())
            .map(|(b, r)| b.checked_mul(*r).map(|v| v / PRECISION))
            .collect()
    }

    /// Quotes the output for `dx` like the variant's `get_amount_out`, with the given `d`.
    fn quote(&self, xp: &[U256], d: U256, i: usize, j: usize, dx: U256) -> Option<U256> {
        if dx.is_zero() {
            return Some(U256::ZERO);
        }
        let x_new = xp[i].checked_add(dx.checked_mul(self.rates[i])? / PRECISION)?;
        let y_new = (self.math.get_y)(i, j, x_new, xp, d, self.amp)?;
        if xp[j] <= y_new {
            return None;
        }
        let offset = if self.math.minus_one_offset { U256::from(1) } else { U256::ZERO };
        let gross = (xp[j] - y_new).checked_sub(offset)?;
        let fee_rate = match self.math.dynamic_fee {
            Some(dynamic_fee) => dynamic_fee(
                xp[i].checked_add(x_new)? / U256::from(2),
                xp[j].checked_add(y_new)? / U256::from(2),
                self.fee,
                self.offpeg_fee_multiplier,
            ),
            None => self.fee,
        };
        if self.math.fee_before_denorm {
            let fee_amount = fee_rate.checked_mul(gross)? / FEE_DENOMINATOR;
            Some(
                gross
                    .checked_sub(fee_amount)?
                    .checked_mul(PRECISION)? /
                    self.rates[j],
            )
        } else {
            let dy = gross.checked_mul(PRECISION)? / self.rates[j];
            let fee_amount = fee_rate.checked_mul(dy)? / FEE_DENOMINATOR;
            dy.checked_sub(fee_amount)
        }
    }

    /// Returns the post-swap `Pool::spot_price` and the swap output for `dx`.
    fn post_swap_price(
        &self,
        xp: &[U256],
        d: U256,
        i: usize,
        j: usize,
        dx: U256,
    ) -> Option<((U256, U256), U256)> {
        let dy = self.quote(xp, d, i, j, dx)?;
        let mut post_balances = self.balances.to_vec();
        post_balances[i] = post_balances[i].checked_add(dx)?;
        post_balances[j] = post_balances[j].checked_sub(dy)?;
        if post_balances[j].is_zero() {
            return None;
        }
        let post_xp = self.xp(&post_balances)?;
        let post_d = (self.math.get_d)(&post_xp, self.amp)?;
        let price = self.price_fraction(&post_xp, &post_balances, post_d, i, j)?;
        Some((price, dy))
    }

    /// Returns the spot price dy/dx, fee included, as `(numerator, denominator)`.
    fn price_fraction(
        &self,
        xp: &[U256],
        balances: &[U256],
        d: U256,
        i: usize,
        j: usize,
    ) -> Option<(U256, U256)> {
        let n = U256::from(xp.len());
        let ann_eff = self.amp.checked_mul(n)? / self.math.a_precision;
        let mut d_p = d;
        for x_k in xp {
            d_p = d_p
                .checked_mul(d)?
                .checked_div(x_k.checked_mul(n)?)?;
        }
        let num_xp = ann_eff
            .checked_mul(xp[i])?
            .checked_add(d_p)?;
        let den_xp = ann_eff
            .checked_mul(xp[j])?
            .checked_add(d_p)?;
        if den_xp.is_zero() {
            return None;
        }
        let effective_fee = match self.math.dynamic_fee {
            Some(dynamic_fee) => dynamic_fee(xp[i], xp[j], self.fee, self.offpeg_fee_multiplier),
            None => self.fee,
        };
        let numerator = num_xp
            .checked_mul(balances[j])?
            .checked_mul(FEE_DENOMINATOR - effective_fee)?;
        let denominator = den_xp
            .checked_mul(balances[i])?
            .checked_mul(FEE_DENOMINATOR)?;
        Some((numerator, denominator))
    }
}

/// Illinois false position on `g(dx) = price(dx) - aim`, with `g(lo) > 0` and `g(hi) < 0`.
fn false_position(
    price_at: impl Fn(U256) -> Option<((U256, U256), U256)>,
    band: &TargetBand,
    mut bracket: Bracket,
) -> Result<U256, SwapToPriceError> {
    let mut best = BestPoint { dx: U256::ZERO, price: f64::INFINITY };
    for _ in 0..MAX_ITERATIONS {
        if bracket.is_closed() {
            break;
        }
        let dx = next_dx(bracket.lo, bracket.hi, bracket.g_lo, bracket.g_hi);
        let Some((price, dy)) = price_at(dx) else {
            bracket.move_hi_unpriced(dx);
            continue;
        };
        let price_f = fraction_to_f64(&price).ok_or(SwapToPriceError::MathFailed)?;
        if fraction_cmp(&price, band.num, band.den) == std::cmp::Ordering::Less {
            bracket.move_hi(dx, price_f - band.aim());
            continue;
        }
        if !dy.is_zero() {
            if price_f <= band.accept_upper() {
                return Ok(dx);
            }
            best = BestPoint { dx, price: price_f };
        }
        bracket.move_lo(dx, price_f - band.aim());
    }
    best.accept(band.target, band.tolerance, bracket.is_closed())
}

/// The target price as an exact fraction and as f64, with the caller's tolerance.
struct TargetBand {
    num: U256,
    den: U256,
    target: f64,
    tolerance: f64,
}

impl TargetBand {
    /// Accepts only the lower half of the band, so caller-side f64 rounding stays in the band.
    fn accept_upper(&self) -> f64 {
        self.target * (1.0 + 0.5 * self.tolerance)
    }

    fn aim(&self) -> f64 {
        self.target * (1.0 + 0.25 * self.tolerance)
    }
}

/// The search interval `[lo, hi]` and the residuals g(lo) > 0 and g(hi) < 0.
struct Bracket {
    lo: U256,
    hi: U256,
    g_lo: f64,
    g_hi: f64,
    last_moved: LastMoved,
}

impl Bracket {
    fn is_closed(&self) -> bool {
        self.hi.saturating_sub(self.lo) <= U256::from(1)
    }

    /// When the same endpoint moves twice in a row, halves the other residual (Illinois step).
    fn move_lo(&mut self, dx: U256, g: f64) {
        if self.last_moved == LastMoved::Lo {
            self.g_hi *= 0.5;
        }
        self.lo = dx;
        self.g_lo = g;
        self.last_moved = LastMoved::Lo;
    }

    fn move_hi(&mut self, dx: U256, g: f64) {
        if self.last_moved == LastMoved::Hi {
            self.g_lo *= 0.5;
        }
        self.hi = dx;
        self.g_hi = g;
        self.last_moved = LastMoved::Hi;
    }

    /// Moves `hi` to an unquotable input. The NaN residual makes `next_dx` bisect.
    fn move_hi_unpriced(&mut self, dx: U256) {
        self.hi = dx;
        self.g_hi = f64::NAN;
        self.last_moved = LastMoved::Neither;
    }
}

/// The largest evaluated input whose post-swap price is at or above the target.
struct BestPoint {
    dx: U256,
    price: f64,
}

impl BestPoint {
    /// Returns non-zero `dx` when its price is in the band, or when the tolerance is zero and the
    /// bracket closed to 1 wei. Otherwise returns [`SwapToPriceError::MathFailed`].
    fn accept(
        &self,
        target: f64,
        tolerance: f64,
        bracket_closed: bool,
    ) -> Result<U256, SwapToPriceError> {
        let in_band = self.price <= target * (1.0 + tolerance);
        let exact_limit = tolerance == 0.0 && bracket_closed;
        if self.dx.is_zero() || !(in_band || exact_limit) {
            return Err(SwapToPriceError::MathFailed);
        }
        Ok(self.dx)
    }
}

/// Compares the fraction `frac.0 / frac.1` against `num / den` without loss of precision.
fn fraction_cmp(frac: &(U256, U256), num: U256, den: U256) -> std::cmp::Ordering {
    let lhs = U512::from(frac.0) * U512::from(den);
    let rhs = U512::from(num) * U512::from(frac.1);
    lhs.cmp(&rhs)
}

fn fraction_to_f64(frac: &(U256, U256)) -> Option<f64> {
    let num = u256_to_f64(frac.0).ok()?;
    let den = u256_to_f64(frac.1).ok()?;
    if den == 0.0 {
        return None;
    }
    Some(num / den)
}

/// Returns the false-position step clamped to `[lo + 1, hi - 1]`, or the midpoint when the
/// residuals give no usable ratio.
fn next_dx(lo: U256, hi: U256, g_lo: f64, g_hi: f64) -> U256 {
    let width = hi - lo;
    let mid = lo + width / U256::from(2);
    if !g_lo.is_finite() || !g_hi.is_finite() || g_lo <= 0.0 || g_hi >= 0.0 {
        return mid;
    }
    let ratio = g_lo / (g_lo - g_hi);
    if !ratio.is_finite() || ratio <= 0.0 || ratio >= 1.0 {
        return mid;
    }
    // f64 keeps a small ratio. A fixed-point ratio rounds it to zero and forces 1 wei steps.
    let Ok(width_f) = u256_to_f64(width) else {
        return mid;
    };
    let Ok(offset) = U256::try_from((width_f * ratio).floor()) else {
        return mid;
    };
    let dx = lo.saturating_add(offset);
    let min_dx = lo + U256::from(1);
    let max_dx = hi - U256::from(1);
    dx.clamp(min_dx, max_dx)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const WAD: u128 = 1_000_000_000_000_000_000;
    const RATE_6_DEC: u128 = 1_000_000_000_000_000_000_000_000_000_000;

    fn v1_two_coin() -> Pool {
        Pool::StableSwapV1 {
            balances: vec![U256::from(50_000_000u128 * WAD), U256::from(48_000_000u128 * WAD)],
            rates: vec![U256::from(WAD), U256::from(WAD)],
            amp: U256::from(2000u64),
            fee: U256::from(1_000_000u64),
        }
    }

    fn v0_two_coin() -> Pool {
        Pool::StableSwapV0 {
            balances: vec![U256::from(50_000_000u128 * WAD), U256::from(48_000_000u128 * WAD)],
            rates: vec![U256::from(WAD), U256::from(WAD)],
            amp: U256::from(2000u64),
            fee: U256::from(1_000_000u64),
        }
    }

    fn v2_two_coin() -> Pool {
        Pool::StableSwapV2 {
            balances: vec![U256::from(50_000_000u128 * WAD), U256::from(48_000_000u128 * WAD)],
            rates: vec![U256::from(WAD), U256::from(WAD)],
            amp: U256::from(200_000u64),
            fee: U256::from(4_000_000u64),
        }
    }

    fn steth_two_coin() -> Pool {
        Pool::StableSwapSTETH {
            balances: vec![U256::from(50_000_000u128 * WAD), U256::from(48_000_000u128 * WAD)],
            rates: vec![U256::from(WAD), U256::from(WAD)],
            amp: U256::from(5000u64),
            fee: U256::from(1_000_000u64),
        }
    }

    fn v1_three_coin_mixed_decimals() -> Pool {
        // 3pool state at block 24669924 (DAI 18 dec, USDC 6 dec, USDT 6 dec).
        Pool::StableSwapV1 {
            balances: vec![
                U256::from(63_975_337_809_806_329_031_583_135u128),
                U256::from(61_219_263_170_093u128),
                U256::from(37_832_425_459_809u128),
            ],
            rates: vec![U256::from(WAD), U256::from(RATE_6_DEC), U256::from(RATE_6_DEC)],
            amp: U256::from(4000u64),
            fee: U256::from(1_500_000u64),
        }
    }

    fn ng_dynamic_fee() -> Pool {
        // The pool is imbalanced so the offpeg fee multiplier raises the fee.
        Pool::StableSwapNG {
            balances: vec![U256::from(1_500_000u128 * WAD), U256::from(700_000u128 * WAD)],
            rates: vec![U256::from(WAD), U256::from(WAD)],
            amp: U256::from(40_000u64),
            fee: U256::from(4_000_000u64),
            offpeg_fee_multiplier: U256::from(20_000_000_000u64),
        }
    }

    fn meta_with_virtual_price() -> Pool {
        // rates[1] carries the base pool's virtual price (1.03).
        Pool::StableSwapMeta {
            balances: vec![U256::from(500_000u128 * WAD), U256::from(480_000u128 * WAD)],
            rates: vec![U256::from(WAD), U256::from(1_030_000_000_000_000_000u128)],
            amp: U256::from(50_000u64),
            fee: U256::from(4_000_000u64),
        }
    }

    fn alend_precision_mul() -> Pool {
        Pool::StableSwapALend {
            balances: vec![
                U256::from(20_000_000u128 * WAD),
                U256::from(18_000_000_000_000u128), // 6 decimals
            ],
            precision_mul: vec![U256::from(1u64), U256::from(1_000_000_000_000u128)],
            amp: U256::from(20_000u64),
            fee: U256::from(2_000_000u64),
            offpeg_fee_multiplier: U256::from(20_000_000_000u64),
        }
    }

    fn spot(pool: &Pool, i: usize, j: usize) -> (U256, U256) {
        pool.spot_price(i, j)
            .expect("spot price")
    }

    fn post_swap_spot(pool: &Pool, i: usize, j: usize, dx: U256) -> (U256, U256) {
        let dy = pool
            .get_amount_out(i, j, dx)
            .expect("get_amount_out");
        let mut post = pool.clone();
        let balances = post.balances().to_vec();
        post.set_balance(i, balances[i] + dx)
            .expect("set balance in");
        post.set_balance(j, balances[j] - dy)
            .expect("set balance out");
        spot(&post, i, j)
    }

    fn scaled_target(spot: &(U256, U256), multiplier: f64) -> (U256, U256) {
        let ppb = U256::from((multiplier * 1e9) as u64);
        (spot.0 * ppb, spot.1 * U256::from(1_000_000_000u64))
    }

    fn assert_in_band(price: &(U256, U256), target: &(U256, U256), tolerance: f64) {
        let price_f = fraction_to_f64(price).expect("failed to convert the post-swap price to f64");
        let target_f = fraction_to_f64(target).expect("failed to convert the target price to f64");
        assert!(price_f >= target_f, "post-swap price {price_f} fell below target {target_f}");
        assert!(
            price_f <= target_f * (1.0 + tolerance),
            "post-swap price {price_f} above tolerance band of target {target_f}"
        );
    }

    #[rstest]
    #[case::v0_two_coin(v0_two_coin(), 0, 1, 0.999)]
    #[case::v0_two_coin_reverse(v0_two_coin(), 1, 0, 0.999)]
    #[case::v2_two_coin(v2_two_coin(), 0, 1, 0.999)]
    #[case::steth_two_coin(steth_two_coin(), 0, 1, 0.999)]
    #[case::v1_two_coin_shallow(v1_two_coin(), 0, 1, 0.9999)]
    #[case::v1_two_coin_deep(v1_two_coin(), 0, 1, 0.999)]
    #[case::v1_two_coin_reverse(v1_two_coin(), 1, 0, 0.999)]
    #[case::v1_mixed_decimals_18_to_6(v1_three_coin_mixed_decimals(), 0, 1, 0.999)]
    #[case::v1_mixed_decimals_6_to_18(v1_three_coin_mixed_decimals(), 1, 0, 0.999)]
    #[case::v1_mixed_decimals_6_to_6(v1_three_coin_mixed_decimals(), 2, 1, 0.9995)]
    #[case::ng_dynamic_fee(ng_dynamic_fee(), 0, 1, 0.999)]
    #[case::ng_dynamic_fee_reverse(ng_dynamic_fee(), 1, 0, 0.999)]
    #[case::meta_virtual_price(meta_with_virtual_price(), 0, 1, 0.999)]
    #[case::meta_virtual_price_reverse(meta_with_virtual_price(), 1, 0, 0.999)]
    #[case::alend_precision_mul(alend_precision_mul(), 0, 1, 0.999)]
    #[case::alend_precision_mul_reverse(alend_precision_mul(), 1, 0, 0.999)]
    fn test_swap_to_price_stableswap_variants(
        #[case] pool: Pool,
        #[case] i: usize,
        #[case] j: usize,
        #[case] multiplier: f64,
    ) {
        let tolerance = 0.001;
        let current = spot(&pool, i, j);
        let target = scaled_target(&current, multiplier);

        let dx = swap_to_price(&pool, i, j, target.0, target.1, tolerance)
            .expect("solver should converge");
        assert!(dx > U256::ZERO, "expected a non-zero swap amount");

        let post = post_swap_spot(&pool, i, j, dx);
        assert_in_band(&post, &target, tolerance);
    }

    /// Fails when the solver's copy of the quote and spot-price rules drifts from `Pool`.
    #[rstest]
    #[case::v0(v0_two_coin())]
    #[case::v1(v1_two_coin())]
    #[case::v1_mixed_decimals(v1_three_coin_mixed_decimals())]
    #[case::v2(v2_two_coin())]
    #[case::steth(steth_two_coin())]
    #[case::ng(ng_dynamic_fee())]
    #[case::meta(meta_with_virtual_price())]
    #[case::alend(alend_precision_mul())]
    fn test_normalized_pool_matches_pool_math(#[case] pool: Pool) {
        let normalized =
            NormalizedStablePool::from_pool(&pool).expect("StableSwap variant is supported");
        let xp = normalized
            .xp(normalized.balances)
            .expect("normalized balances");
        let d = (normalized.math.get_d)(&xp, normalized.amp).expect("invariant");
        let n = pool.balances().len();
        for i in 0..n {
            for j in (0..n).filter(|&j| j != i) {
                assert_eq!(
                    normalized.price_fraction(&xp, normalized.balances, d, i, j),
                    pool.spot_price(i, j),
                    "spot price differs for {i} -> {j}"
                );
                for divisor in [1_000_000u64, 1_000, 10] {
                    let dx = pool.balances()[i] / U256::from(divisor);
                    let quote = normalized.quote(&xp, d, i, j, dx);
                    assert_eq!(
                        quote,
                        pool.get_amount_out(i, j, dx),
                        "quote differs for {i} -> {j}"
                    );
                    let (price, dy) = normalized
                        .post_swap_price(&xp, d, i, j, dx)
                        .expect("post-swap price");
                    assert_eq!(Some(dy), quote);
                    assert_eq!(price, post_swap_spot(&pool, i, j, dx), "post-swap price differs");
                }
            }
        }
    }

    #[rstest]
    #[case::v1(v1_two_coin())]
    #[case::ng(ng_dynamic_fee())]
    fn test_swap_to_price_target_above_spot(#[case] pool: Pool) {
        let current = spot(&pool, 0, 1);
        let target = scaled_target(&current, 1.01);
        let result = swap_to_price(&pool, 0, 1, target.0, target.1, 0.001);
        assert_eq!(result, Err(SwapToPriceError::TargetAboveSpot));
    }

    #[test]
    fn test_swap_to_price_target_below_limit() {
        let pool = v1_two_coin();
        let current = spot(&pool, 0, 1);
        let target = (current.0, current.1 * U256::from(100u64));
        let result = swap_to_price(&pool, 0, 1, target.0, target.1, 0.001);
        assert_eq!(result, Err(SwapToPriceError::TargetBelowLimit));
    }

    #[rstest]
    #[case::same_coin(0, 0, U256::from(1u64), U256::from(1u64))]
    #[case::index_out_of_range(0, 2, U256::from(1u64), U256::from(1u64))]
    #[case::zero_numerator(0, 1, U256::ZERO, U256::from(1u64))]
    #[case::zero_denominator(0, 1, U256::from(1u64), U256::ZERO)]
    fn test_swap_to_price_invalid_input(
        #[case] i: usize,
        #[case] j: usize,
        #[case] target_num: U256,
        #[case] target_den: U256,
    ) {
        let result = swap_to_price(&v1_two_coin(), i, j, target_num, target_den, 0.001);
        assert!(
            matches!(result, Err(SwapToPriceError::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }

    #[rstest]
    #[case::in_band(7, 1.0005, 0.001, false, Ok(U256::from(7u64)))]
    #[case::zero_tolerance_closed_bracket(7, 1.0005, 0.0, true, Ok(U256::from(7u64)))]
    #[case::zero_tolerance_open_bracket(7, 1.0005, 0.0, false, Err(SwapToPriceError::MathFailed))]
    #[case::above_band(7, 1.01, 0.001, true, Err(SwapToPriceError::MathFailed))]
    #[case::zero_dx_in_band(0, 1.0, 0.001, true, Err(SwapToPriceError::MathFailed))]
    fn test_best_point_accept(
        #[case] dx: u64,
        #[case] price: f64,
        #[case] tolerance: f64,
        #[case] bracket_closed: bool,
        #[case] expected: Result<U256, SwapToPriceError>,
    ) {
        let best = BestPoint { dx: U256::from(dx), price };
        assert_eq!(best.accept(1.0, tolerance, bracket_closed), expected);
    }

    #[test]
    fn test_next_dx_small_ratio_on_wide_bracket() {
        let hi = U256::from(1u64) << 80;
        let dx = next_dx(U256::ZERO, hi, 1e-12, -1.0);
        let expected = 2f64.powi(80) * 1e-12;
        let dx_f = u256_to_f64(dx).expect("failed to convert dx to f64");
        assert!(
            ((dx_f - expected) / expected).abs() < 1e-9,
            "next_dx gave {dx_f}, expected about {expected}"
        );
    }

    #[test]
    fn test_swap_to_price_target_equal_to_spot() {
        let pool = v1_two_coin();
        let current = spot(&pool, 0, 1);
        let dx = swap_to_price(&pool, 0, 1, current.0, current.1, 0.001).expect("equal target");
        assert_eq!(dx, U256::ZERO);
    }

    #[test]
    fn test_swap_to_price_crypto_variant() {
        let wad = U256::from(WAD);
        let pool = Pool::TwoCryptoNG {
            balances: [U256::from(5000u64) * wad, U256::from(5000u64) * wad],
            precisions: [U256::from(1u64), U256::from(1u64)],
            price_scale: wad,
            d: U256::from(10000u64) * wad,
            ann: U256::from(540_000u64) * U256::from(10_000u64),
            gamma: U256::from(11_809_167_828_997u64),
            mid_fee: U256::from(3_000_000u64),
            out_fee: U256::from(30_000_000u64),
            fee_gamma: U256::from(230_000_000_000_000u64),
        };
        let result = swap_to_price(&pool, 0, 1, U256::from(1u64), U256::from(2u64), 0.001);
        assert_eq!(result, Err(SwapToPriceError::UnsupportedVariant));
    }

    #[test]
    fn test_swap_to_price_zero_tolerance() {
        let pool = v1_two_coin();
        let current = spot(&pool, 0, 1);
        let target = scaled_target(&current, 0.999);
        let dx = swap_to_price(&pool, 0, 1, target.0, target.1, 0.0)
            .expect("swap_to_price failed with zero tolerance");
        assert!(dx > U256::ZERO);
        let post = post_swap_spot(&pool, 0, 1, dx);
        assert_ne!(
            fraction_cmp(&post, target.0, target.1),
            std::cmp::Ordering::Less,
            "best-effort result must not undershoot the target"
        );
        assert_in_band(&post, &target, 1e-6);
    }
}
