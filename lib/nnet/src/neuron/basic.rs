use crate::{Activate, ActivationFunction, Neuron, NeuronActivate};

/// A basic neuron.
#[derive(Debug, PartialEq)]
pub struct Basic {
    /// Shifts the neuron's overall sensitivity.
    bias: f64,

    /// The weights the neuron applies to its inputs.
    weights: Vec<f64>,

    /// The activation function to use.
    activation: ActivationFunction,
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
    /// use nnet::{Neuron, BasicNeuron, ActivationFunction};
    ///
    /// let neuron = BasicNeuron::builder()
    ///  .bias(0.0)
    ///  .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///  .activation(ActivationFunction::linear())
    ///  .build();
    /// ```
    #[must_use]
    pub fn builder() -> Builder {
        Builder::default()
    }
}

impl Neuron {
    #[must_use]
    pub fn basic() -> Builder {
        Builder::default()
    }
}

impl From<Basic> for Neuron {
    fn from(basic: Basic) -> Neuron {
        Neuron::Basic(basic)
    }
}

impl NeuronActivate for Basic {
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
///   .activation(ActivationFunction::linear())
///  .build();
/// ```
#[derive(Default)]
pub struct Builder {
    bias: f64,
    weights: Vec<f64>,
    activation: Option<ActivationFunction>,
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
    ///   .activation(ActivationFunction::linear())
    ///   .build();
    /// ```
    #[must_use]
    pub fn bias(mut self, bias: f64) -> Self {
        self.bias = bias;
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
    ///  .activation(ActivationFunction::linear())
    ///  .build();
    /// ```
    #[must_use]
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = weights;
        self
    }

    /// Add a weight for the neuron.
    ///
    /// # Arguments
    ///
    /// - `weight` is multiplied against each of the input values.
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
    ///     .bias(0.0)
    ///     .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///     .activation(ActivationFunction::linear())
    ///     .build();
    /// ```
    pub fn add_weight(mut self, weight: f64) -> Self {
        self.weights.push(weight);
        self
    }

    /// Add multiple weights for the neuron.
    ///
    /// # Arguments
    ///
    /// - `weight` is multiplied against each of the input values.
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
    ///    .bias(0.0)
    ///    .weights(vec![0.1, 0.2, 0.3, 0.4])
    ///    .activation(ActivationFunction::linear())
    ///    .build();
    /// ```
    pub fn add_weights(mut self, weight: Vec<f64>) -> Self {
        self.weights.extend(weight);
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
    ///   .activation(ActivationFunction::linear())
    ///   .build();
    /// ```
    #[must_use]
    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        self.activation = Some(activation);
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
    ///   .activation(ActivationFunction::linear())
    ///   .build();
    /// ```
    pub fn build(self) -> Basic {
        Basic {
            bias: self.bias,
            weights: self.weights,
            activation: self.activation.unwrap_or_else(ActivationFunction::sigmoid),
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
            .activation(ActivationFunction::linear())
            .build();

        let inputs = [0.1, 0.2, 0.3, 0.4];
        let expected = 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4;
        let output = neuron.activate(&inputs);
        assert!(
            (output - expected).abs() < f64::EPSILON,
            "Expected {output} to be close to {expected}",
        );
    }
}
