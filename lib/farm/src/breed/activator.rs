use crate::{
    genome::activator::{Gene, Genome},
    mutate::Mutator,
};
use evo::Breed;
use rand::prelude::Distribution;

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
/// let left = Genome::new(Gene::Linear);
/// let right = Genome::new(Gene::Sigmoid);
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
    fn crossover(&self, (left, right): (&Genome, &Genome)) -> Genome {
        let activator = if self.mutator.check_mutate() {
            left.activator()
        } else {
            right.activator()
        };

        Genome::new(activator.clone())
    }

    fn mutate(&self, mut genome: Genome) -> Genome {
        if self.mutator.mutation_size() > 0.0 && self.mutator.check_mutate() {
            let af = rand::random::<Gene>();
            genome.set_activator(af);
        }

        genome
    }
}

impl Distribution<Gene> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Gene {
        match rng.gen_range(0..2) {
            0 => Gene::Linear,
            _ => Gene::Sigmoid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossover() {
        let mutator = Mutator::builder().mutation_rate(1.0).build();

        let left = Genome::new(Gene::Linear);
        let right = Genome::new(Gene::Sigmoid);

        let breeder = Breeder::new(mutator);
        let offspring = breeder.crossover((&left, &right));

        assert_eq!(offspring.activator(), &Gene::Linear);

        let offspring = breeder.crossover((&right, &left));

        assert_eq!(offspring.activator(), &Gene::Sigmoid);
    }
}
