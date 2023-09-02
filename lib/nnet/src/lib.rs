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
pub mod network;
pub mod neuron;

pub use crate::{
    activation::{Activate, Function as ActivationFunction},
    layer::Layer,
    network::Network,
    neuron::{Activate as NeuronActivate, Basic as BasicNeuron, Neuron},
};
