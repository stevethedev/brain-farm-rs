use super::Crossover;
use crate::mutate::Target;
use rand::distributions::Standard;
use rand::prelude::Distribution;

/// The gene for an activation function.
///
/// # Examples
///
/// ```
/// use farm::genome::activator::Gene;
///
/// let gene = Gene::Linear;
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Genome {
    pub activator: Gene,
}

/// Enable crossover for [`Genome`].
///
/// # Examples
///
/// ```
/// use farm::genome::{Crossover, activator::{Genome, Gene}};
///
/// let left = Genome { activator: Gene::Linear };
/// let right = Genome { activator: Gene::Sigmoid };
///
/// let target = left.crossover(&right);
/// ```
impl Crossover for Genome {
    fn crossover(&self, other: &Self) -> Self {
        Self {
            activator: self.activator.crossover(&other.activator),
        }
    }
}

/// The gene for an activation function.
#[derive(Clone, Debug, PartialEq)]
pub enum Gene {
    /// Linear activation function.
    Linear,

    /// Sigmoid activation function.
    Sigmoid,
}

impl Distribution<Gene> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Gene {
        match rng.gen_range(0..2) {
            0 => Gene::Linear,
            _ => Gene::Sigmoid,
        }
    }
}

/// Enable crossover for [`Gene`].
///
/// # Examples
///
/// ```
/// use farm::genome::{Crossover, activator::Gene};
///
/// let left = Gene::Linear;
/// let right = Gene::Sigmoid;
///
/// let target = left.crossover(&right);
/// ```
impl Crossover for Gene {
    fn crossover(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Linear, Self::Linear) => Self::Linear,
            (Self::Sigmoid, Self::Sigmoid) => Self::Sigmoid,
            _ => {
                if rand::random() {
                    self.clone()
                } else {
                    other.clone()
                }
            }
        }
    }
}

/// Ensures that the gene can be bred.
///
/// # Examples
///
/// ```
/// use farm::{
///     genome::activator::Gene,
///     mutate::{Mutator, Target},
/// };
///
/// let mutator = Mutator::builder().build();
///
/// let gene = Gene::Linear;
/// let gene = mutator.mutate(gene);
/// ```
impl Target for Gene {
    fn mutate(mut self, mutator: &crate::mutate::Mutator) -> Self {
        if mutator.mutation_size() > 0.0 && mutator.check_mutate() {
            self = rand::random::<Gene>();
        }

        self
    }
}
