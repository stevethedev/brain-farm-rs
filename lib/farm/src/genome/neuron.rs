use super::activator;
use crate::genome::{Crossover, Generate};
use crate::mutate::{Mutator, Target, VecMutation};

/// Genome for a neuron.
///
/// # Examples
///
/// ```
/// use farm::genome::{activator, neuron::Genome};
///
/// let activator = activator::Genome { activator: activator::Gene::Linear };
/// let weights = vec![0.0, 1.0, 2.0];
/// let bias = 3.0;
/// let genome = Genome { activator: activator.clone(), weights: weights.clone(), bias };
/// assert_eq!(genome.activator, activator);
/// assert_eq!(genome.weights, weights);
/// assert_eq!(genome.bias, bias);
/// ```
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Genome {
    pub activator: activator::Genome,
    pub weights: Vec<Gene>,
    pub bias: Gene,
}

/// Configuration for generating a [`Genome`].
///
/// # Examples
///
/// ```
/// use farm::genome::{neuron::{Genome, GenerateConfig}, activator};
/// use farm::genome::Generate;
///
/// let activator_generator = || activator::Genome { activator: activator::Gene::Linear };
/// let weight_generator = || vec![0.0, 1.0, 2.0];
/// let bias_generator = || 3.0;
///
/// let config = GenerateConfig {
///     activator_generator,
///     weight_generator,
///     bias_generator,
/// };
///
/// let genome = Genome::generate(&config);
///
/// assert_eq!(genome.activator.activator, activator::Gene::Linear);
/// assert_eq!(genome.weights, vec![0.0, 1.0, 2.0]);
/// assert_eq!(genome.bias, 3.0);
/// ```
pub struct GenerateConfig<TActivatorGenerator, TWeightGenerator, TBiasGenerator>
where
    TActivatorGenerator: Fn() -> activator::Genome,
    TWeightGenerator: Fn() -> Vec<Gene>,
    TBiasGenerator: Fn() -> Gene,
{
    pub activator_generator: TActivatorGenerator,
    pub weight_generator: TWeightGenerator,
    pub bias_generator: TBiasGenerator,
}

impl<TActivatorGenerator, TWeightGenerator, TBiasGenerator>
    Generate<&GenerateConfig<TActivatorGenerator, TWeightGenerator, TBiasGenerator>> for Genome
where
    TActivatorGenerator: Fn() -> activator::Genome,
    TWeightGenerator: Fn() -> Vec<Gene>,
    TBiasGenerator: Fn() -> Gene,
{
    fn generate(
        config: &GenerateConfig<TActivatorGenerator, TWeightGenerator, TBiasGenerator>,
    ) -> Self {
        let activator = (config.activator_generator)();
        let weights = (config.weight_generator)();
        let bias = (config.bias_generator)();

        Self {
            activator,
            weights,
            bias,
        }
    }
}

/// Enable crossover for [`Genome`].
///
/// # Examples
///
/// ```
/// use farm::genome::{Crossover, neuron::Genome, activator};
///
/// let left = Genome {
///     activator: activator::Genome { activator: activator::Gene::Linear },
///     weights: vec![],
///     bias: 0.0,
/// };
/// let right = Genome {
///     activator: activator::Genome { activator: activator::Gene::Sigmoid },
///     weights: vec![],
///     bias: 0.0,
/// };
///
/// let target = left.crossover(&right);
/// ```
impl Crossover for Genome {
    fn crossover(&self, other: &Self) -> Self {
        Self {
            activator: self.activator.crossover(&other.activator),
            weights: self.weights.crossover(&other.weights),
            bias: self.bias.crossover(&other.bias),
        }
    }
}

impl Target for Genome {
    /// Enable mutation for [`Genome`].
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::genome::{neuron::Genome, activator};
    /// use farm::mutate::{Mutator, Target};
    ///
    /// let mutator = Mutator::builder().build();
    ///
    /// let activator = activator::Genome { activator: activator::Gene::Linear };
    /// let weights = vec![0.0, 1.0, 2.0];
    /// let bias = 3.0;
    /// let genome = Genome { activator: activator.clone(), weights: weights.clone(), bias };
    /// let genome = genome.mutate(&mutator);
    /// ```
    fn mutate(mut self, mutator: &Mutator) -> Self {
        self.activator = mutator.mutate(self.activator);
        self.weights = mutator.mutate(self.weights);
        self.bias = mutator.mutate(self.bias);

        // Transposition mutation swaps two weights.
        if mutator.check_mutate() {
            mutate_weights(&mut self.weights);
        }

        self
    }
}

/// Mutate the weights of a neuron.
///
/// # Arguments
///
/// - `weights` - The weights to mutate.
fn mutate_weights(weights: &mut Vec<Gene>) {
    let (min, max) = weights
        .iter()
        .filter(|n| n.is_finite() && !n.is_nan())
        .fold((None, None), |(min, max), n| {
            let min = if let Some(min) = min { n.min(min) } else { *n };
            let max = if let Some(max) = max { n.max(max) } else { *n };
            (Some(min), Some(max))
        });

    if let (Some(min), Some(max)) = (min, max) {
        if min < max {
            let vec_mutation = VecMutation::new(weights.len(), || f64::generate(min..=max));
            if matches!(
                vec_mutation,
                VecMutation::Swap(_, _) | VecMutation::Replace(_, _) | VecMutation::Reverse(_, _)
            ) {
                vec_mutation.apply(weights);
            }
        }
    };
}

pub type Gene = f64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize() {
        let activator = activator::Genome {
            activator: activator::Gene::Linear,
        };
        let weights = vec![0.0, 1.0, 2.0];
        let bias = 3.0;
        let genome = Genome {
            activator: activator.clone(),
            weights: weights.clone(),
            bias,
        };
        let serialized = serde_json::to_string(&genome).unwrap();
        let expected = r#"{"activator":{"activator":"Linear"},"weights":[0.0,1.0,2.0],"bias":3.0}"#;

        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_deserialize() {
        let activator = activator::Genome {
            activator: activator::Gene::Linear,
        };
        let weights = vec![0.0, 1.0, 2.0];
        let bias = 3.0;
        let genome = Genome {
            activator: activator.clone(),
            weights: weights.clone(),
            bias,
        };
        let serialized = r#"
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
        "#;
        let deserialized: Genome = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, genome);
    }
}
