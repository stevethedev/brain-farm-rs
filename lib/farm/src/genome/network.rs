use super::layer;
use crate::genome::{Create, Crossover, Extract, Generate};
use crate::mutate::Target;
use nnet::Network;

/// A neural network genome.
///
/// # Examples
///
/// ```
/// use farm::genome::{network, layer, neuron, activator, Generate};
///
/// let neuron_config = neuron::GenerateConfig {
///     activator_generator: || activator::Genome::generate(()),
///     weight_generator: || std::iter::from_fn(|| Some(f64::generate(-1.0..=1.0))).take(3).collect(),
///     bias_generator: || f64::generate(-2.0..=2.0),
/// };
///
/// let layer_config = layer::GenerateConfig {
///     neuron_generator: || std::iter::from_fn(|| Some(neuron::Genome::generate(&neuron_config))).take(4).collect(),
/// };
///
/// let network_config = network::GenerateConfig {
///     layer_generator: || std::iter::from_fn(|| Some(layer::Genome::generate(&layer_config))).take(5).collect(),
/// };
///
/// let genome = network::Genome::generate(&network_config);
///
/// assert_eq!(genome.layers.len(), 5);
/// assert_eq!(genome.layers[0].neurons.len(), 4);
/// assert_eq!(genome.layers[0].neurons[0].weights.len(), 3);
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Genome {
    pub layers: Vec<layer::Genome>,
}

/// Configuration for generating a [`Genome`].
///
/// # Examples
///
/// ```
/// use farm::genome::{network, layer, neuron, activator, Generate};
///
/// let neuron_config = neuron::GenerateConfig {
///     activator_generator: || activator::Genome::generate(()),
///     weight_generator: || std::iter::from_fn(|| Some(f64::generate(-1.0..=1.0))).take(3).collect(),
///     bias_generator: || f64::generate(-2.0..=2.0),
/// };
///
/// let layer_config = layer::GenerateConfig {
///     neuron_generator: || std::iter::from_fn(|| Some(neuron::Genome::generate(&neuron_config))).take(5).collect(),
/// };
///
/// let network_config = network::GenerateConfig {
///     layer_generator: || std::iter::from_fn(|| Some(layer::Genome::generate(&layer_config))).take(5).collect(),
/// };
///
/// let genome = network::Genome::generate(&network_config);
///
/// assert_eq!(genome.layers.len(), 5);
/// assert_eq!(genome.layers[0].neurons.len(), 5);
/// assert_eq!(genome.layers[0].neurons[0].weights.len(), 3);
/// ```
pub struct GenerateConfig<TLayerGenerator>
where
    TLayerGenerator: Fn() -> Vec<layer::Genome>,
{
    pub layer_generator: TLayerGenerator,
}

impl<TLayerGenerator> Generate<&GenerateConfig<TLayerGenerator>> for Genome
where
    TLayerGenerator: Fn() -> Vec<layer::Genome>,
{
    fn generate(config: &GenerateConfig<TLayerGenerator>) -> Self {
        let layers = (config.layer_generator)();

        Self { layers }
    }
}

/// Ensures that the genome can be bred.
///
/// # Examples
///
/// ```
/// use farm::{
///    genome::network::{Genome, GenerateConfig},
///   mutate::{Mutator, Target},
/// };
/// use farm::genome::{Crossover, Generate};
///
/// let mutator = Mutator::builder().build();
///
/// let left = Genome::generate(&GenerateConfig {
///     layer_generator: || vec![],
/// });
/// let right = Genome::generate(&GenerateConfig {
///     layer_generator: || vec![],
/// });
///
/// let target = left.crossover(&right);
/// ```
impl Crossover for Genome {
    fn crossover(&self, other: &Self) -> Self {
        let layers = Vec::crossover(&self.layers, &other.layers);
        Self { layers }
    }
}

/// Enable mutation for [`Genome`].
///
/// # Examples
///
/// ```
/// use farm::{
///     genome::network::{Genome, GenerateConfig},
///     mutate::{Mutator, Target},
/// };
/// use farm::genome::Generate;
///
/// let mutator = Mutator::builder().build();
///
/// let genome = Genome::generate(&GenerateConfig {
///     layer_generator: || vec![],
/// });
///
/// let genome = genome.mutate(&mutator);
/// ```
impl Target for Genome {
    fn mutate(mut self, mutator: &crate::mutate::Mutator) -> Self {
        self.layers = self.layers.mutate(mutator);
        // TODO: mutate the network layer vector.
        self
    }
}

impl Create<Network> for Genome {
    /// Create a new [`Network`] from the genome.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::genome::{network, layer, neuron, activator, Generate, Create};
    ///
    /// let neuron_config = neuron::GenerateConfig {
    ///     activator_generator: || activator::Genome::generate(()),
    ///     weight_generator: || std::iter::from_fn(|| Some(f64::generate(-1.0..=1.0))).take(3).collect(),
    ///     bias_generator: || f64::generate(-2.0..=2.0),
    /// };
    ///
    /// let layer_config = layer::GenerateConfig {
    ///     neuron_generator: || std::iter::from_fn(|| Some(neuron::Genome::generate(&neuron_config))).take(5).collect(),
    /// };
    ///
    /// let network_config = network::GenerateConfig {
    ///     layer_generator: || std::iter::from_fn(|| Some(layer::Genome::generate(&layer_config))).take(5).collect(),
    /// };
    ///
    /// let genome = network::Genome::generate(&network_config);
    /// let network = genome.create();
    /// ```
    fn create(&self) -> Network {
        let layers = self.layers.iter().map(layer::Genome::create).collect();
        Network::builder().layers(layers).build()
    }
}

impl Extract<Genome> for Network {
    /// Extract a [`Genome`] from the [`Network`].
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::genome::{network, layer, neuron, activator, Generate, Extract, Create};
    ///
    /// let neuron_config = neuron::GenerateConfig {
    ///     activator_generator: || activator::Genome::generate(()),
    ///     weight_generator: || std::iter::from_fn(|| Some(f64::generate(-1.0..=1.0))).take(3).collect(),
    ///     bias_generator: || f64::generate(-2.0..=2.0),
    /// };
    ///
    /// let layer_config = layer::GenerateConfig {
    ///     neuron_generator: || std::iter::from_fn(|| Some(neuron::Genome::generate(&neuron_config))).take(5).collect(),
    /// };
    ///
    /// let network_config = network::GenerateConfig {
    ///     layer_generator: || std::iter::from_fn(|| Some(layer::Genome::generate(&layer_config))).take(5).collect(),
    /// };
    ///
    /// let genome = network::Genome::generate(&network_config);
    /// let network = genome.create();
    ///
    /// let genome = network.genome();
    ///
    /// assert_eq!(genome.layers.len(), 5);
    /// assert_eq!(genome.layers[0].neurons.len(), 5);
    /// assert_eq!(genome.layers[0].neurons[0].weights.len(), 3);
    /// ```
    fn genome(&self) -> Genome {
        let layers = self.layers().iter().map(nnet::Layer::genome).collect();
        Genome { layers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize() {
        let genome = Genome {
            layers: vec![layer::Genome { neurons: vec![] }],
        };

        let serialized = serde_json::to_string(&genome).unwrap();
        let expected = r#"{"layers":[{"neurons":[]}]}"#;

        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_deserialize() {
        let genome = Genome {
            layers: vec![layer::Genome { neurons: vec![] }],
        };

        let serialized = r#"
            {
                "layers": [
                    {
                        "neurons": []
                    }
                ]
            }
        "#;
        let deserialized: Genome = serde_json::from_str(serialized).unwrap();

        assert_eq!(genome, deserialized);
    }
}
