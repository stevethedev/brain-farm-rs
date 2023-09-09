mod basic;

pub use basic::{Basic, Builder as BasicNeuronBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum Neuron {
    Basic(Basic),
}

impl Neuron {
    /// Get the neuron's activation function.
    ///
    /// # Returns
    ///
    /// The activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Neuron, BasicNeuron, ActivationFunction};
    ///
    /// let neuron = BasicNeuron::builder()
    ///     .bias(0.0)
    ///     .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///     .activation(ActivationFunction::linear())
    ///     .build();
    ///
    /// assert_eq!(neuron.activation(), &ActivationFunction::linear());
    /// ```
    #[must_use]
    pub fn activator(&self) -> &crate::ActivationFunction {
        match self {
            Self::Basic(basic) => basic.activation(),
        }
    }

    /// Get the neuron's bias.
    ///
    /// # Returns
    ///
    /// The bias.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Neuron, BasicNeuron, ActivationFunction};
    ///
    /// let neuron = BasicNeuron::builder()
    ///     .bias(0.0)
    ///     .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///     .activation(ActivationFunction::linear())
    ///     .build();
    ///
    /// assert_eq!(neuron.bias(), 0.0);
    /// ```
    #[must_use]
    pub fn bias(&self) -> f64 {
        match self {
            Self::Basic(basic) => basic.bias(),
        }
    }

    /// Get the neuron's weights.
    ///
    /// # Returns
    ///
    /// The weights.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Neuron, BasicNeuron, ActivationFunction};
    ///
    /// let neuron = BasicNeuron::builder()
    ///     .bias(0.0)
    ///     .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///     .activation(ActivationFunction::linear())
    ///     .build();
    ///
    /// assert_eq!(neuron.weights(), &[0.1, 0.2, 0.3, 0.4]);
    /// ```
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        match self {
            Self::Basic(basic) => basic.weights(),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize() {
        let neuron = Neuron::Basic(Basic::builder().build());
        let serialized = serde_json::to_string(&neuron).unwrap();
        let expected = r#"{"Basic":{"bias":0.0,"weights":[],"activation":{"Sigmoid":null}}}"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_deserialize() {
        let serialized = r#"{"Basic":{"bias":0.0,"weights":[],"activation":{"Sigmoid":null}}}"#;
        let deserialized: Neuron = serde_json::from_str(serialized).unwrap();
        let expected = Basic::builder().build();
        assert_eq!(deserialized, Neuron::Basic(expected));
    }
}
