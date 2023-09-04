use super::activator;
use crate::genome::Crossover;
use crate::mutate::{Mutator, Target};

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
pub struct Genome {
    pub activator: activator::Genome,
    pub weights: Vec<Gene>,
    pub bias: Gene,
}

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
    fn mutate(mut self, mutator: &Mutator) -> Self {
        self.activator = mutator.mutate(self.activator);
        self.weights = mutator.mutate(self.weights);
        self.bias = mutator.mutate(self.bias);

        self
    }
}

pub type Gene = f64;
