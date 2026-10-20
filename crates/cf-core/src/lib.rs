pub mod cfwm;
pub mod dexp;
pub mod fma;
pub mod rcp;
mod rcp_table;
pub mod scalar_kernels;
pub mod sha256;
pub mod simd;
#[cfg(target_arch = "x86_64")]
pub mod simd_avx2;
#[cfg(target_arch = "x86_64")]
pub mod simd_scalar_fma;
pub mod workspace;

#[cfg(feature = "soft-fma")]
pub mod fma_check;

pub use fma::{fma, mul_add2, FmaOps, NativeFma};
pub use rcp::{rcp_ps, rcp_ps_bits, RCP_TABLE, RCP_TABLE_U32};
pub use simd::{avx2_available, ActConsts, Backend, EchoActConsts, ScalarSimd, Simd};
#[cfg(target_arch = "x86_64")]
pub use simd_avx2::Avx2Simd;
#[cfg(target_arch = "x86_64")]
pub use simd_scalar_fma::ScalarFmaSimd;
pub use workspace::{Arena, Slot, Workspace, MAX_SLOTS};
