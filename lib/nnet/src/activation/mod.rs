mod linear;
mod sigmoid;

pub use linear::Linear;
pub use sigmoid::Sigmoid;

pub type DynActivationFunction = Box<dyn ActivationFunction>;

/// Neuron activation function
///
/// This trait is implemented by the various activation functions that can be used by neurons.
pub trait ActivationFunction {
    fn activate(&self, x: f64) -> f64;
}
