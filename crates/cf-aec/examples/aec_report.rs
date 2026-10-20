//! Prints the measured linear-AEC results used in `README.md`.
//!
//! Run with:
//!
//! ```text
//! cargo run -p cf-aec --release --example aec_report
//! ```
//!
//! This is a functional report, not a bit-exactness check: the linear stage has
//! no reference implementation to match. The scenario code lives in the shared
//! validation harness so the report and the tests cannot drift apart.

#[path = "../tests/common/mod.rs"]
mod common;

fn main() {
    // The report keeps several `Aec` instances alive at once and each one owns
    // its fixed-size histories (about 170 KiB). The default Windows main-thread
    // stack is 1 MiB, which this report sits close to; run it on a roomy worker
    // thread so the documented command cannot overflow the stack.
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(common::print_report)
        .expect("spawn the report thread")
        .join()
        .expect("join the report thread");
}
