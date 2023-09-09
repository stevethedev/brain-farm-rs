use super::Crossover;
use crate::genome::{Create, Extract};
use crate::mutate::{Mutator, Target};
use nnet::ActivationFunction;
use rand::distributions::{Distribution, Standard};

/// The gene for an activation function.
///
/// # Examples
///
/// ```
/// use farm::genome::activator::Gene;
///
/// let gene = Gene::Linear;
/// ```
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Genome {
    pub activator: Gene,
}

impl super::Generate<()> for Genome {
    /// Create a new genome with a random activation function.
    ///
    /// # Returns
    ///
    /// The new genome.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::genome::activator::Genome;
    /// use farm::genome::Generate;
    ///
    /// let genome = Genome::generate(());
    /// ```
    fn generate(empty: ()) -> Self {
        Self::generate(&empty)
    }
}

impl super::Generate<&()> for Genome {
    /// Create a new genome with a random activation function.
    ///
    /// # Returns
    ///
    /// The new genome.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::genome::activator::Genome;
    /// use farm::genome::Generate;
    ///
    /// let genome = Genome::generate(&());
    /// ```
    fn generate(_: &()) -> Self {
        Self::generate(rand::random::<Gene>)
    }
}

impl<F> super::Generate<F> for Genome
where
    F: Fn() -> Gene,
{
    /// Create a new genome with a random activation function.
    ///
    /// # Returns
    ///
    /// The new genome.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::genome::activator::{Genome, Gene};
    /// use farm::genome::Generate;
    ///
    /// let genome = Genome::generate(|| Gene::Sigmoid);
    /// assert_eq!(genome.activator, Gene::Sigmoid);
    /// ```
    fn generate(activator_generator: F) -> Self {
        Self {
            activator: activator_generator(),
        }
    }
}

/// Ensures that the genome can be bred.
///
/// # Examples
///
/// ```
/// use farm::{
///     genome::activator::{Gene, Genome},
///     mutate::{Mutator, Target},
/// };
///
/// let mutator = Mutator::builder().build();
///
/// let genome = Genome { activator: Gene::Linear };
/// let genome = genome.mutate(&mutator);
/// ```
impl Target for Genome {
    fn mutate(mut self, mutator: &Mutator) -> Self {
        self.activator = self.activator.mutate(mutator);
        self
    }
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

/// Enable creation for [`Genome`].
///
/// # Examples
///
/// ```
/// use farm::genome::{activator::{Genome, Gene}, Create};
/// use nnet::ActivationFunction;
///
/// let genome = Genome { activator: Gene::Linear };
/// let activator = genome.create();
///
/// assert_eq!(activator, ActivationFunction::linear());
/// ```
impl Create<ActivationFunction> for Genome {
    fn create(&self) -> ActivationFunction {
        match self.activator {
            Gene::Linear => ActivationFunction::linear(),
            Gene::Sigmoid => ActivationFunction::sigmoid(),
        }
    }
}

/// Enable extraction for [`Genome`].
///
/// # Examples
///
/// ```
/// use farm::genome::{activator::{Genome, Gene}, Extract};
/// use nnet::ActivationFunction;
///
/// let genome = ActivationFunction::linear().genome();
///
/// assert_eq!(genome, Genome { activator: Gene::Linear });
/// ```
impl Extract<Genome> for ActivationFunction {
    fn genome(&self) -> Genome {
        let activator = match self {
            Self::Linear(_) => Gene::Linear,
            Self::Sigmoid(_) => Gene::Sigmoid,
        };

        Genome { activator }
    }
}

/// The gene for an activation function.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize() {
        let genome = Genome {
            activator: Gene::Linear,
        };

        let serialized = serde_json::to_string(&genome).unwrap();
        assert_eq!(serialized, r#"{"activator":"Linear"}"#);
    }

    #[test]
    fn test_deserialize() {
        let genome = Genome {
            activator: Gene::Linear,
        };

        let deserialized: Genome = serde_json::from_str(r#"{"activator":"Linear"}"#).unwrap();
        assert_eq!(deserialized, genome);
    }
}
