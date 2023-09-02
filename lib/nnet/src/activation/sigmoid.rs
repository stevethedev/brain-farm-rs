use super::{Activate, Function};

/// Sigmoid activation function
///
/// This function is used to squash the output of a neuron to a value between 0 and 1.
#[derive(Debug, PartialEq)]
pub struct Sigmoid;

impl Activate for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        let n_exp = (-x).exp();
        1.0 / (1.0 + n_exp)
    }
}

impl Function {
    /// Sigmoid activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::ActivationFunction;
    ///
    /// let sig = ActivationFunction::sigmoid();
    /// ```
    #[must_use]
    pub fn sigmoid() -> Self {
        Self::Sigmoid(Sigmoid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let expected: f64 = 1.0
            / (1.0
                + std::f64::consts::E.powf(-(0.1 * 0.1 + 0.5 * 0.2 + 1.0 * 0.3 + 1.5 * 0.4 + 1.0)));

        let sig = Sigmoid;
        let outputs = sig.activate(0.1 * 0.1 + 0.5 * 0.2 + 1.0 * 0.3 + 1.5 * 0.4 + 1.0);
        assert!(
            (outputs - expected).abs() < f64::EPSILON,
            "Expected {outputs} to be close to {expected}"
        );
    }
}
