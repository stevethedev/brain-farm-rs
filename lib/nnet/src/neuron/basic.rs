use super::Neuron;
use crate::{ActivationFunction, DynActivationFunction};

/// A basic neuron.
pub struct Basic {
    /// Shifts the neuron's overall sensitivity.
    bias: f64,

    /// The weights the neuron applies to its inputs.
    weights: Vec<f64>,

    /// The activation function to use.
    activation: DynActivationFunction,
}

impl Basic {
    /// Create a new neuron builder.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{BasicNeuron, ActivationFunction};
    ///
    /// let neuron = BasicNeuron::builder()
    ///  .bias(0.0)
    ///  .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///  .activation(ActivationFunction::Linear)
    ///  .build();
    /// ```
    pub fn builder() -> Builder {
        Builder::default()
    }
}

impl Neuron for Basic {
    fn activate(&self, inputs: &[f64]) -> f64 {
        let sum = sum(&self.weights, inputs, self.bias);
        self.activation.activate(sum)
    }
}

/// Sum the products of the weights and inputs.
///
/// # Arguments
///
/// - `weights` are multiplied against each of the input values.
/// - `inputs` are multiplied against the weights.
/// - `bias` is added to the sum.
///
/// # Returns
///
/// The sum of the products of the weights and inputs.
fn sum(weights: &[f64], inputs: &[f64], bias: f64) -> f64 {
    let product = Iterator::zip(weights.iter(), inputs.iter())
        .map(|(weight, input)| weight * input)
        .sum::<f64>();

    product + bias
}

/// A builder for `Basic` neurons.
///
/// # Examples
///
/// ```
/// use nnet::{BasicNeuron, ActivationFunction};
///
/// let neuron = BasicNeuron::builder()
///    .bias(0.0)
///   .weights(vec![0.1, 0.2, 0.3, 0.4])
///   .activation(ActivationFunction::Linear)
///  .build();
/// ```
#[derive(Default)]
pub struct Builder {
    bias: Option<f64>,
    weights: Option<Vec<f64>>,
    activation: Option<DynActivationFunction>,
}

impl Builder {
    /// Set the bias for the neuron.
    ///
    /// # Arguments
    ///
    /// - `bias` is added to the sum.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{BasicNeuron, ActivationFunction};
    ///
    /// let neuron = BasicNeuron::builder()
    ///   .bias(0.0)
    ///   .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///   .activation(ActivationFunction::Linear)
    ///   .build();
    /// ```
    pub fn bias(mut self, bias: f64) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Set the weights for the neuron.
    ///
    /// # Arguments
    ///
    /// - `weights` are multiplied against each of the input values.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{BasicNeuron, ActivationFunction};
    ///
    /// let neuron = BasicNeuron::builder()
    ///  .bias(0.0)
    ///  .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///  .activation(ActivationFunction::Linear)
    ///  .build();
    /// ```
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set the activation function for the neuron.
    ///
    /// # Arguments
    ///
    /// - `activation` function to use.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{BasicNeuron, ActivationFunction};
    ///
    /// let neuron = BasicNeuron::builder()
    ///   .bias(0.0)
    ///   .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///   .activation(ActivationFunction::Linear)
    ///   .build();
    /// ```
    pub fn activation<F: ActivationFunction + 'static>(mut self, activation: F) -> Self {
        self.activation = Some(Box::new(activation));
        self
    }

    /// Build the neuron.
    ///
    /// # Returns
    ///
    /// The neuron.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{BasicNeuron, ActivationFunction};
    ///
    /// let neuron = BasicNeuron::builder()
    ///   .bias(0.0)
    ///   .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///   .activation(ActivationFunction::Linear)
    ///   .build();
    /// ```
    pub fn build(self) -> Basic {
        Basic {
            bias: self.bias.unwrap_or(0.0),
            weights: self.weights.unwrap_or_default(),
            activation: self
                .activation
                .unwrap_or_else(|| Box::new(crate::activation::Sigmoid)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let neuron = Builder::default()
            .bias(0.0)
            .weights(vec![0.1, 0.2, 0.3, 0.4])
            .activation(crate::activation::Linear)
            .build();

        let inputs = [0.1, 0.2, 0.3, 0.4];
        let expected = 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4;
        let output = neuron.activate(&inputs);
        assert!(
            (output - expected).abs() < f64::EPSILON,
            "Expected {} to be close to {}",
            output,
            expected
        );
    }
}
