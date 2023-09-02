mod basic;

pub use basic::{Basic as BasicNeuron, Builder as BasicNeuronBuilder};

/// Neuron trait
///
/// This trait is implemented by the various types of neurons that can be used by networks.
pub trait Neuron {
    /// Activate the neuron
    fn activate(&self, inputs: &[f64]) -> f64;
}
