mod basic;

pub use basic::{Basic, Builder as BasicNeuronBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
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
