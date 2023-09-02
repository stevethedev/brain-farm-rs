#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic
)]

mod activation;
mod neuron;

pub use activation::*;
pub use neuron::*;
