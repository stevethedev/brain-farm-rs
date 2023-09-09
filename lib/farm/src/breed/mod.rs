use crate::{
    genome::Crossover,
    mutate::{Mutator, Target},
};
pub use evo::Breed;

/// Breeds activation functions.
///
/// # Examples
///
/// ```
/// use farm::{
///     breed::{Breed, Breeder},
///     mutate::Mutator,
///     genome::activator::{Genome, Gene},
/// };
///
/// let mutator = Mutator::builder().build();
///
/// let left = Genome { activator: Gene::Linear };
/// let right = Genome { activator: Gene::Sigmoid };
///
/// let breeder = Breeder::new(mutator);
/// let offspring = breeder.crossover((&left, &right));
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
    #[must_use]
    pub fn new(mutator: Mutator) -> Self {
        Self { mutator }
    }
}

impl<TGenome> Breed<TGenome> for Breeder
where
    TGenome: Crossover + Target,
{
    /// Breed offspring from two parents.
    ///
    /// # Arguments
    ///
    /// - `pair` - The parents to breed.
    ///
    /// # Returns
    ///
    /// The offspring.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::{
    ///     breed::{Breed, Breeder},
    ///     genome::{
    ///         activator::{Gene as ActivatorGene, Genome as ActivatorGenome},
    ///         layer::{Gene as LayerGene, Genome as LayerGenome},
    ///         neuron::{Gene as NeuronGene, Genome as NeuronGenome},
    ///     },
    ///     mutate::{Mutator, Target},
    /// };
    ///
    /// let mutator = Mutator::builder().build();
    ///
    /// let left = LayerGenome { neurons: vec![] };
    /// let right = LayerGenome { neurons: vec![] };
    ///
    /// let breeder = Breeder::new(mutator);
    /// let offspring = breeder.crossover((&left, &right));
    /// ```
    fn crossover(&self, pair: (&TGenome, &TGenome)) -> TGenome {
        pair.0.crossover(pair.1)
    }

    /// Mutate the genome.
    ///
    /// # Arguments
    ///
    /// - `genome` - The genome to mutate.
    ///
    /// # Returns
    ///
    /// The mutated genome.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::{
    ///     breed::{Breed, Breeder},
    ///     mutate::{Mutator, Target},
    ///     genome::activator::{Genome, Gene},
    /// };
    ///
    /// let mutator = Mutator::builder().build();
    ///
    /// let genome = Genome { activator: Gene::Linear };
    /// let genome = mutator.mutate(genome);
    /// ```
    fn mutate(&self, genome: TGenome) -> TGenome {
        genome.mutate(&self.mutator)
    }
}
