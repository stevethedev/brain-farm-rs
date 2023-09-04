use super::Breed;
use crate::{
    genome::{neuron::Genome, Crossover},
    mutate::Mutator,
};

/// Breed neuron genomes.
///
/// # Examples
///
/// ```
/// use evo::Breed;
/// use farm::{
///     breed::{neuron::Breeder},
///     genome::{
///         activator::{Gene as ActivatorGene, Genome as ActivatorGenome},
///         neuron::{Gene as NeuronGene, Genome as NeuronGenome},
///     },
///     mutate::{Mutator, Target},
/// };
///
/// let mutator = Mutator::builder().build();
///
/// let activator = ActivatorGenome { activator: ActivatorGene::Linear };
/// let weights = vec![0.0, 1.0, 2.0];
/// let bias = 3.0;
/// let genome = NeuronGenome { activator: activator.clone(), weights: weights.clone(), bias };
///
/// let breeder = Breeder::new(mutator);
/// let genome = breeder.mutate(genome);
/// ```
pub struct Breeder {
    mutator: Mutator,
}

impl Breeder {
    /// Create a new breeder.
    ///
    /// # Arguments
    ///
    /// - `mutator` to mutate the offspring with.
    ///
    /// # Returns
    ///
    /// The new breeder.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::{
    ///     breed::{neuron::Breeder},
    ///     mutate::Mutator,
    /// };
    ///
    /// let mutator = Mutator::builder().build();
    /// let breeder = Breeder::new(mutator);
    /// ```
    pub fn new(mutator: Mutator) -> Self {
        Self { mutator }
    }
}

/// Breed neuron genomes.
///
/// # Examples
///
/// ```
/// use farm::{
///     breed::{neuron::Breeder, Breed},
///     genome::{
///         activator::{Gene as ActivatorGene, Genome as ActivatorGenome},
///         neuron::{Gene as NeuronGene, Genome as NeuronGenome},
///     },
///     mutate::{Mutator, Target},
/// };
///
/// let mutator = Mutator::builder().build();
///
/// let activator = ActivatorGenome { activator: ActivatorGene::Linear };
/// let weights = vec![0.0, 1.0, 2.0];
/// let bias = 3.0;
/// let genome = NeuronGenome { activator: activator.clone(), weights: weights.clone(), bias };
///
/// let breeder = Breeder::new(mutator);
/// let genome = breeder.mutate(genome);
/// ```
impl Breed<Genome> for Breeder {
    fn crossover(&self, (left, right): (&Genome, &Genome)) -> Genome {
        Genome::crossover(left, right)
    }

    fn mutate(&self, genome: Genome) -> Genome {
        self.mutator.mutate(genome)
    }
}
