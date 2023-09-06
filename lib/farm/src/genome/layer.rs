use super::neuron;
use crate::genome::Generate;
use crate::{
    genome::Crossover,
    mutate::{Mutator, Target},
};

/// Genome for a layer.
///
/// # Examples
///
/// ```
/// use farm::genome::{neuron, activator, layer};
///
/// let neurons = vec![
///     neuron::Genome {
///         activator: activator::Genome {
///             activator: activator::Gene::Linear
///         },
///         weights: vec![0.0, 1.0, 2.0],
///         bias: 3.0,
///    },
/// ];
/// let genome = layer::Genome { neurons: neurons.clone() };
/// assert_eq!(genome.neurons, neurons);
/// ```
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Genome {
    pub neurons: Gene,
}

/// Configuration for generating a [`Genome`].
///
/// # Examples
///
/// ```
/// use rand::{thread_rng, Rng};
/// use farm::genome::{neuron, activator, layer};
/// use farm::genome::Generate;
///
/// let activator_generator = || activator::Genome::generate(());
/// let weight_generator = || std::iter::from_fn(|| Some(f64::generate(-1.0..=1.0))).take(3).collect();
/// let bias_generator = || f64::generate(-2.0..=2.0);
///
/// let neuron_config = neuron::GenerateConfig {
///     activator_generator,
///     weight_generator,
///     bias_generator,
/// };
///
/// let neuron_generator = || std::iter::from_fn(|| Some(neuron::Genome::generate(&neuron_config))).take(5).collect();
/// let layer_config = layer::GenerateConfig { neuron_generator };
///
/// let genome = layer::Genome::generate(&layer_config);
///
/// assert_eq!(genome.neurons.len(), 5);
/// assert_eq!(genome.neurons[0].weights.len(), 3);
/// ```
pub struct GenerateConfig<TNeuronGenerator>
where
    TNeuronGenerator: Fn() -> Gene,
{
    pub neuron_generator: TNeuronGenerator,
}

impl<TNeuronGenerator> Generate<&GenerateConfig<TNeuronGenerator>> for Genome
where
    TNeuronGenerator: Fn() -> Gene,
{
    fn generate(config: &GenerateConfig<TNeuronGenerator>) -> Self {
        let neurons = (config.neuron_generator)();

        Self { neurons }
    }
}

/// Enable crossover for [`Genome`].
///
/// # Examples
///
/// ```
/// use farm::genome::{Crossover, layer::Genome};
///
/// let left = Genome { neurons: vec![] };
/// let right = Genome { neurons: vec![] };
///
/// let target = left.crossover(&right);
/// ```
impl Crossover for Genome {
    fn crossover(&self, other: &Self) -> Self {
        Self {
            neurons: Vec::crossover(&self.neurons, &other.neurons),
        }
    }
}

/// Enable mutation for [`Genome`].
///
/// # Examples
///
/// ```
/// use farm::genome::{layer::Genome, neuron, activator};
/// use farm::mutate::{Mutator, Target};
///
/// let mutator = Mutator::builder().build();
///
/// let neurons = vec![
///     neuron::Genome {
///         activator: activator::Genome { activator: activator::Gene::Linear },
///         weights: vec![0.0, 1.0, 2.0],
///         bias: 3.0,
///     },
/// ];
/// let genome = Genome { neurons: neurons.clone() };
/// let genome = mutator.mutate(genome);
/// ```
impl Target for Genome {
    fn mutate(mut self, mutator: &Mutator) -> Self {
        self.neurons = mutator.mutate(self.neurons);
        // TODO: Support structural mutations on the neurons.
        self
    }
}

pub type Gene = Vec<neuron::Genome>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::activator;

    #[test]
    fn test_serialize() {
        let genome = Genome {
            neurons: vec![neuron::Genome {
                activator: activator::Genome {
                    activator: activator::Gene::Linear,
                },
                weights: vec![0.0, 1.0, 2.0],
                bias: 3.0,
            }],
        };

        let serialized = r#"{"neurons":[{"activator":{"activator":"Linear"},"weights":[0.0,1.0,2.0],"bias":3.0}]}"#;

        assert_eq!(serde_json::to_string(&genome).unwrap(), serialized);
    }

    #[test]
    fn test_deserialize() {
        let genome = Genome {
            neurons: vec![neuron::Genome {
                activator: activator::Genome {
                    activator: activator::Gene::Linear,
                },
                weights: vec![0.0, 1.0, 2.0],
                bias: 3.0,
            }],
        };

        let deserialized: Genome = serde_json::from_str(
            r#"{
                "neurons": [
                    {
                        "activator": {
                            "activator": "Linear"
                        },
                        "weights": [
                            0.0,
                            1.0,
                            2.0
                        ],
                        "bias": 3.0
                    }
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(genome, deserialized);
    }
}
