use crate::Layer;
use serde::{Deserialize, Serialize};

/// A neural network.
///
/// # Examples
///
/// ```
/// use nnet::{Network, Layer, BasicNeuron};
///
/// let neuron = BasicNeuron::builder().build();
/// let layer = Layer::builder().add_neuron(neuron).build();
/// let network = Network::builder().add_layer(layer).build();
///
/// let inputs = vec![0.1, 0.2, 0.3, 0.4];
/// let outputs = network.activate(&inputs);
///
/// assert_eq!(outputs.len(), 1);
/// ```
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    /// Create a new builder.
    ///
    /// # Returns
    ///
    /// A new builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::Network;
    ///
    /// let network = Network::builder().build();
    /// ```
    #[must_use]
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Activate the network.
    ///
    /// # Arguments
    ///
    /// - `inputs` to activate the network with.
    ///
    /// # Returns
    ///
    /// The output of the network.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Network, Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    /// let network = Network::builder().add_layer(layer).build();
    ///
    /// let inputs = vec![0.1, 0.2, 0.3, 0.4];
    /// let outputs = network.activate(&inputs);
    ///
    /// assert_eq!(outputs.len(), 1);
    /// ```
    #[must_use]
    pub fn activate(&self, inputs: &[f64]) -> Vec<f64> {
        self.layers
            .iter()
            .fold(inputs.to_vec(), |values, layer| layer.activate(&values))
    }

    /// Get a reference to the set of layers.
    ///
    /// # Returns
    ///
    /// A reference to the set of layers.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Network, Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    /// let network = Network::builder().add_layer(layer).build();
    ///
    /// assert_eq!(network.layers().len(), 1);
    /// ```
    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Parse a JSON string into a network.
    ///
    /// # Arguments
    ///
    /// - `json` is the JSON string to parse.
    ///
    /// # Returns
    ///
    /// The parsed network.
    ///
    /// # Errors
    ///
    /// If the network cannot be parsed.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Network, Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    /// let network = Network::builder().add_layer(layer).build();
    ///
    /// let serialized = serde_json::to_string(&network).unwrap();
    ///
    /// let parsed = Network::parse_json(&serialized).unwrap();
    ///
    /// assert_eq!(network, parsed);
    /// ```
    pub fn parse_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize the network to a JSON string.
    ///
    /// # Returns
    ///
    /// The serialized network.
    ///
    /// # Errors
    ///
    /// If the network cannot be serialized.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Network, Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    /// let network = Network::builder().add_layer(layer).build();
    ///
    /// let serialized = network.to_json().unwrap();
    ///
    /// let parsed = Network::parse_json(&serialized).unwrap();
    ///
    /// assert_eq!(network, parsed);
    /// ```
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// A builder for [`Network`].
///
/// # Examples
///
/// ```
/// use nnet::{Network, Layer, BasicNeuron};
///
/// let neuron = BasicNeuron::builder().build();
/// let layer = Layer::builder().add_neuron(neuron).build();
/// let network = Network::builder().add_layer(layer).build();
/// ```
#[derive(Default)]
pub struct Builder {
    layers: Vec<Layer>,
}

impl Builder {
    /// Add a layer to the network.
    ///
    /// # Arguments
    ///
    /// - `layer` to add to the network.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Network, Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    /// let network = Network::builder().add_layer(layer).build();
    ///
    /// assert_eq!(network.layers().len(), 1);
    /// ```
    #[must_use]
    pub fn add_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Add multiple layers to the network.
    ///
    /// # Arguments
    ///
    /// - `layers` to add to the network.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Network, Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    /// let network = Network::builder().add_layers(vec![layer]).build();
    ///
    /// assert_eq!(network.layers().len(), 1);
    /// ```
    #[must_use]
    pub fn add_layers(mut self, layers: Vec<Layer>) -> Self {
        self.layers.extend(layers);
        self
    }

    /// Build the network.
    ///
    /// # Returns
    ///
    /// The network.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Network, Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    /// let network = Network::builder().add_layer(layer).build();
    /// ```
    #[must_use]
    pub fn build(self) -> Network {
        Network {
            layers: self.layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_network() {
        assert_eq!(Network::builder().build(), Network { layers: vec![] });
        assert_eq!(
            Network::builder()
                .add_layer(Layer::builder().build())
                .build(),
            Network {
                layers: vec![Layer::builder().build()]
            }
        );
        assert_eq!(
            Network::builder()
                .add_layer(Layer::builder().build())
                .add_layer(Layer::builder().build())
                .build(),
            Network {
                layers: vec![Layer::builder().build(), Layer::builder().build()]
            }
        );
    }
}
