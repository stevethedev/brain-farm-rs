#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic
)]

mod activation;
mod layer;
mod neuron;

pub use activation::*;
pub use layer::*;
pub use neuron::*;
