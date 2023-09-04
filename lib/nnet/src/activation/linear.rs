use super::{Activate, Function};
use serde::{Deserialize, Serialize};

/// Linear activation function.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Linear;

impl Activate for Linear {
    fn activate(&self, input: f64) -> f64 {
        input
    }
}

impl Function {
    /// Linear activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::ActivationFunction;
    ///
    /// let lin = ActivationFunction::linear();
    /// ```
    #[must_use]
    pub fn linear() -> Self {
        Self::Linear(Linear)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let expected: f64 = 0.1 * 0.1 + 0.5 * 0.2 + 1.0 * 0.3 + 1.5 * 0.4 + 1.0;

        let lin = Linear;
        let outputs = lin.activate(0.1 * 0.1 + 0.5 * 0.2 + 1.0 * 0.3 + 1.5 * 0.4 + 1.0);
        assert!(
            (outputs - expected).abs() < f64::EPSILON,
            "Expected {outputs} to be close to {expected}"
        );
    }

    #[test]
    fn test_serialize() {
        let lin = Linear;
        let serialized = serde_json::to_string(&lin).unwrap();
        let expected = r#"null"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_deserialize() {
        let lin = Linear;
        let deserialized = serde_json::from_str(r#"null"#).unwrap();
        assert_eq!(lin, deserialized);
    }
}
