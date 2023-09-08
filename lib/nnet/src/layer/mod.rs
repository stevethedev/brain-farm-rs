use crate::{Neuron, NeuronActivate};
use serde::{Deserialize, Serialize};

/// A layer of neurons.
///
/// # Examples
///
/// ```
/// use nnet::{Layer, BasicNeuron};
///
/// let neuron = BasicNeuron::builder().build();
/// let layer = Layer::builder().add_neuron(neuron).build();
/// ```
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Create a new builder.
    ///
    /// # Returns
    ///
    /// A new builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::Layer;
    ///
    /// let layer = Layer::builder().build();
    /// ```
    #[must_use]
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Activate the layer.
    ///
    /// # Arguments
    ///
    /// - `inputs` to activate the layer with.
    ///
    /// # Returns
    ///
    /// The output of the layer.
    #[must_use]
    pub fn activate(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|n| n.activate(inputs)).collect()
    }

    /// Get a reference to the set of neurons.
    ///
    /// # Returns
    ///
    /// A reference to the set of neurons.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    ///
    /// assert_eq!(layer.neurons().len(), 1);
    /// ```
    #[must_use]
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }
}

/// A builder for `Layer`s.
///
/// # Examples
///
/// ```
/// use nnet::{Layer, BasicNeuron};
///
/// let neuron = BasicNeuron::builder().build();
/// let layer = Layer::builder().add_neuron(neuron).build();
/// ```
#[derive(Default)]
pub struct Builder {
    neurons: Vec<Neuron>,
}

impl Builder {
    /// Add a neuron to the layer.
    ///
    /// # Arguments
    ///
    /// - `neuron` to add to the layer.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    ///
    /// assert_eq!(layer.neurons().len(), 1);
    /// ```
    #[must_use]
    pub fn add_neuron(mut self, neuron: impl Into<Neuron>) -> Self {
        self.neurons.push(neuron.into());
        self
    }

    /// Add a vector of neurons to the layer.
    ///
    /// # Arguments
    ///
    /// - `neurons` to add to the layer.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Layer, Neuron};
    ///
    /// let neuron = Neuron::basic().build().into();
    /// let layer = Layer::builder().add_neurons(vec![neuron]).build();
    ///
    /// assert_eq!(layer.neurons().len(), 1);
    /// ```
    #[must_use]
    pub fn add_neurons(mut self, neurons: Vec<Neuron>) -> Self {
        self.neurons.extend(neurons);
        self
    }

    /// Set the neurons for the layer.
    ///
    /// # Arguments
    ///
    /// - `neurons` to set for the layer.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Layer, Neuron};
    ///
    /// let neuron = Neuron::basic().build().into();
    /// let layer = Layer::builder().neurons(vec![neuron]).build();
    ///
    /// assert_eq!(layer.neurons().len(), 1);
    /// ```
    #[must_use]
    pub fn neurons(mut self, neurons: Vec<Neuron>) -> Self {
        self.neurons = neurons;
        self
    }

    /// Build the layer.
    ///
    /// # Returns
    ///
    /// The built layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::Layer;
    ///
    /// let layer = Layer::builder().build();
    /// ```
    #[must_use]
    pub fn build(self) -> Layer {
        let Self { neurons } = self;
        Layer { neurons }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ActivationFunction;

    #[test]
    fn test_layer() {
        let layer = Layer::builder()
            .add_neuron(
                Neuron::basic()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(ActivationFunction::sigmoid())
                    .build(),
            )
            .add_neuron(
                Neuron::basic()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(ActivationFunction::sigmoid())
                    .build(),
            )
            .build();

        let output = layer.activate(&[0.0, 0.0]);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_serialize() {
        let layer = Layer::builder()
            .add_neuron(
                Neuron::basic()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(ActivationFunction::sigmoid())
                    .build(),
            )
            .add_neuron(
                Neuron::basic()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(ActivationFunction::sigmoid())
                    .build(),
            )
            .build();

        let serialized = serde_json::to_string(&layer).unwrap();
        let expected = r#"{"neurons":[{"Basic":{"bias":0.0,"weights":[0.0,0.0],"activation":{"Sigmoid":null}}},{"Basic":{"bias":0.0,"weights":[0.0,0.0],"activation":{"Sigmoid":null}}}]}"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_deserialize() {
        let layer = Layer::builder()
            .add_neuron(
                Neuron::basic()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(ActivationFunction::sigmoid())
                    .build(),
            )
            .add_neuron(
                Neuron::basic()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(ActivationFunction::sigmoid())
                    .build(),
            )
            .build();

        let deserialized: Layer = serde_json::from_str(
            r#"{"neurons":[{"Basic":{"bias":0.0,"weights":[0.0,0.0],"activation":{"Sigmoid":null}}},{"Basic":{"bias":0.0,"weights":[0.0,0.0],"activation":{"Sigmoid":null}}}]}"#,
        )
        .unwrap();
        assert_eq!(layer, deserialized);
    }
}
