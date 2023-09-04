mod linear;
mod sigmoid;

pub use linear::Linear;
use serde::{Deserialize, Serialize};
pub use sigmoid::Sigmoid;

/// [`Neuron`] activation function.
///
/// # Examples
///
/// ```
/// use nnet::ActivationFunction;
///
/// let lin = ActivationFunction::linear();
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Function {
    Linear(Linear),
    Sigmoid(Sigmoid),
}

impl Activate for Function {
    /// Activate the function.
    ///
    /// # Arguments
    ///
    /// - `input` to activate the function with.
    ///
    /// # Returns
    ///
    /// The output of the function.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::ActivationFunction;
    ///
    /// let lin = ActivationFunction::linear();
    fn activate(&self, input: f64) -> f64 {
        match self {
            Self::Linear(lin) => lin.activate(input),
            Self::Sigmoid(sig) => sig.activate(input),
        }
    }
}

/// Trait for executing an activation function.
pub trait Activate {
    /// Activate the function.
    ///
    /// # Arguments
    ///
    /// - `input` to activate the function with.
    ///
    /// # Returns
    ///
    /// The output of the function.
    fn activate(&self, input: f64) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize() {
        let lin = Function::linear();
        let serialized = serde_json::to_string(&lin).unwrap();
        let expected = r#"{"Linear":null}"#;
        assert_eq!(serialized, expected);

        let sig = Function::sigmoid();
        let serialized = serde_json::to_string(&sig).unwrap();
        let expected = r#"{"Sigmoid":null}"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_deserialize() {
        let lin = Function::linear();
        let deserialized: Function = serde_json::from_str(r#"{"Linear":null}"#).unwrap();
        assert_eq!(lin, deserialized);

        let sig = Function::sigmoid();
        let deserialized: Function = serde_json::from_str(r#"{"Sigmoid":null}"#).unwrap();
        assert_eq!(sig, deserialized);
    }
}
