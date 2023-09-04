use crate::genome::Crossover;
use crate::{
    genome::activator::Genome,
    mutate::{Mutator, Target},
};
use evo::Breed;

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

/// Breeds activation functions.
///
/// # Examples
///
/// ```
/// use farm::{
///     breed::{Breed, activator::Breeder},
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
    pub fn new(mutator: Mutator) -> Self {
        Self { mutator }
    }
}

impl Breed<Genome> for Breeder {
    /// Breed offspring from two parents.
    ///
    /// # Arguments
    ///
    /// - `parents` - The parents to breed.
    ///
    /// # Returns
    ///
    /// The offspring.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::{
    ///     breed::{Breed, activator::Breeder},
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
    fn crossover(&self, parents: (&Genome, &Genome)) -> Genome {
        Genome::crossover(parents.0, parents.1)
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
    ///     breed::{Breed, activator::Breeder},
    ///     mutate::{Mutator, Target},
    ///     genome::activator::{Genome, Gene},
    /// };
    ///
    /// let mutator = Mutator::builder().build();
    ///
    /// let genome = Genome { activator: Gene::Linear };
    /// let genome = mutator.mutate(genome);
    /// ```
    fn mutate(&self, genome: Genome) -> Genome {
        genome.mutate(&self.mutator)
    }
}
