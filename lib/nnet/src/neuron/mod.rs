mod basic;

pub use basic::{Basic as BasicNeuron, Builder as BasicNeuronBuilder};

/// Neuron trait
///
/// This trait is implemented by the various types of neurons that can be used by networks.
pub trait Neuron {
    /// Activate the neuron
    fn activate(&self, inputs: &[f64]) -> f64;
}

impl<N: Neuron + 'static> From<N> for Box<dyn Neuron> {
    fn from(neuron: N) -> Box<dyn Neuron> {
        Box::new(neuron)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_trait() {
        struct TestNeuron(f64);

        impl Neuron for TestNeuron {
            fn activate(&self, inputs: &[f64]) -> f64 {
                self.0
            }
        }

        let neuron = TestNeuron(1.0);
        let boxed_neuron: Box<dyn Neuron> = Box::from(neuron);

        assert_eq!(boxed_neuron.activate(&[]), 1.0);
    }
}
