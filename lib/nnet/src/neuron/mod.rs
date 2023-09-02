mod basic;

pub use basic::{Basic, Builder as BasicNeuronBuilder};

#[derive(Debug, PartialEq)]
pub enum Neuron {
    Basic(Basic),
}

/// Neuron trait
///
/// This trait is implemented by the various types of neurons that can be used by networks.
pub trait Activate {
    fn activate(&self, inputs: &[f64]) -> f64;
}

impl Activate for Neuron {
    fn activate(&self, inputs: &[f64]) -> f64 {
        match self {
            Self::Basic(basic) => basic.activate(inputs),
        }
    }
}
