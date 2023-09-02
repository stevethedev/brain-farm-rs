#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic
)]

pub mod activation;
pub mod layer;
pub mod neuron;

pub use crate::{
    activation::{ActivationFunction, DynActivationFunction},
    layer::Layer,
    neuron::{BasicNeuron, Neuron},
};
