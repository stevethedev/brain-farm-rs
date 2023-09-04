use super::{inject_genomes, sort_generation, unrank_generation, Tournament};
use crate::{Breed, BreedManager, CompareRecord, FitnessCalc, Generation, Predict};

/// Runs the genetic algorithm.
pub struct Run<TGenome, TBreeder>
where
    TGenome: Predict + PartialOrd,
    TBreeder: Breed<TGenome>,
{
    breeder: BreedManager<TGenome, TBreeder>,
    fitness_calc: FitnessCalc,
    elitism: usize,
    tournament_size: usize,
}

impl<TGenome, TBreeder> Run<TGenome, TBreeder>
where
    TGenome: Predict + PartialOrd,
    TBreeder: Breed<TGenome>,
{
    /// Creates a new builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::Rng;
    /// use evo::{Breed, Compare, CompareRecord, FitnessCalc, Predict, EvoAlgorithm, TrainingRecord};
    ///
    /// #[derive(Debug, PartialEq, PartialOrd, Clone)]
    /// struct Genome {
    ///     value: f64,
    /// }
    ///
    /// impl Predict for Genome {
    ///     fn predict(&self, input: &[f64]) -> Vec<f64> {
    ///         input.iter().map(|x| x * self.value).collect()
    ///     }
    /// }
    ///
    /// struct Comparator;
    ///
    /// impl Compare<Genome> for Comparator {
    ///     fn compare(
    ///         &self,
    ///         left: &CompareRecord<&Genome>,
    ///         right: &CompareRecord<&Genome>,
    ///     ) -> std::cmp::Ordering {
    ///         left.fitness.partial_cmp(&right.fitness).unwrap()
    ///     }
    /// }
    ///
    /// struct Breeder {
    ///     mut_range: f64,
    /// }
    ///
    /// impl Breed<Genome> for Breeder {
    ///     fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
    ///         Genome {
    ///             value: (pair.0.value + pair.1.value) / 2.0,
    ///         }
    ///     }
    ///
    ///     fn mutate(&self, mut genome: Genome) -> Genome {
    ///         genome.value += rand::thread_rng().gen_range(-1.0..=1.0) * self.mut_range;
    ///         genome
    ///     }
    /// }
    ///
    /// let fitness_calc = FitnessCalc::builder()
    ///     .add_training_record(TrainingRecord {
    ///         input: vec![0.0, 0.0],
    ///         output: vec![0.0],
    ///     })
    ///     .build();
    ///
    /// let breeder = Breeder { mut_range: 0.1 };
    /// let comparator = Comparator;
    /// let algo = EvoAlgorithm::builder()
    ///     .breeder(breeder)
    ///     .fitness_calc(fitness_calc)
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder() -> Builder<TGenome, TBreeder> {
        Builder::default()
    }

    /// Runs the genetic algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::Rng;
    /// use evo::{Breed, Compare, CompareRecord, FitnessCalc, Predict, EvoAlgorithm, TrainingRecord};
    ///
    /// #[derive(Debug, PartialEq, PartialOrd, Clone)]
    /// struct Genome {
    ///     value: f64,
    /// }
    ///
    /// impl Predict for Genome {
    ///     fn predict(&self, input: &[f64]) -> Vec<f64> {
    ///         input.iter().map(|x| x * self.value).collect()
    ///     }
    /// }
    ///
    /// struct Comparator;
    ///
    /// impl Compare<Genome> for Comparator {
    ///     fn compare(
    ///         &self,
    ///         left: &CompareRecord<&Genome>,
    ///         right: &CompareRecord<&Genome>,
    ///     ) -> std::cmp::Ordering {
    ///         left.fitness.partial_cmp(&right.fitness).unwrap()
    ///     }
    /// }
    ///
    /// struct Breeder {
    ///     mut_range: f64,
    /// }
    ///
    /// impl Breed<Genome> for Breeder {
    ///     fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
    ///         Genome {
    ///             value: (pair.0.value + pair.1.value) / 2.0,
    ///         }
    ///     }
    ///
    ///     fn mutate(&self, mut genome: Genome) -> Genome {
    ///         genome.value += rand::thread_rng().gen_range(-1.0..=1.0) * self.mut_range;
    ///         genome
    ///     }
    /// }
    ///
    /// let fitness_calc = FitnessCalc::builder()
    ///     .add_training_record(TrainingRecord {
    ///         input: vec![0.0, 0.0],
    ///         output: vec![0.0],
    ///     })
    ///     .build();
    ///
    /// let breeder = Breeder { mut_range: 0.1 };
    /// let comparator = Comparator;
    /// let algo = EvoAlgorithm::builder()
    ///     .breeder(breeder)
    ///     .fitness_calc(fitness_calc)
    ///     .build()
    ///     .unwrap();
    ///
    /// let generation = vec![
    ///     Genome { value: 1.0 },
    ///     Genome { value: 2.0 },
    ///     Genome { value: 3.0 },
    /// ];
    ///
    /// let actual = algo.run(generation.clone());
    ///
    /// assert_eq!(actual.len(), generation.len());
    /// assert_ne!(actual, generation);
    /// ```
    pub fn run(&self, generation: Generation<TGenome>) -> Generation<TGenome> {
        let ranked_generation = self.rank_generation(generation);

        let next_generation = self.breed_generation(&ranked_generation);
        let elite = self.partition_elite(ranked_generation);

        inject_genomes(next_generation, elite)
    }

    /// Breeds a new generation of genomes.
    ///
    /// # Arguments
    ///
    /// - `parent_generation`: The parent generation to breed.
    ///
    /// # Returns
    ///
    /// A new generation of genomes.
    fn breed_generation(&self, parent_generation: &[CompareRecord<TGenome>]) -> Vec<TGenome> {
        let next_generation = self.new_generation(parent_generation);
        unrank_generation(next_generation)
    }

    /// Partitions the elite genomes from the generation.
    ///
    /// # Arguments
    ///
    /// - `ranked_generation`: The ranked generation to partition.
    ///
    /// # Returns
    ///
    /// The elite genomes.
    fn partition_elite(&self, ranked_generation: Vec<CompareRecord<TGenome>>) -> Vec<TGenome> {
        let sorted_generation = sort_generation(ranked_generation);
        let sorted_generation = unrank_generation(sorted_generation);

        let elitism = std::cmp::min(self.elitism, sorted_generation.len());
        let mut elite = sorted_generation;
        elite.truncate(elitism);
        elite
    }

    /// Ranks the generation.
    ///
    /// # Arguments
    ///
    /// - `generation`: The generation to rank.
    ///
    /// # Returns
    ///
    /// The ranked generation.
    fn rank_generation(&self, generation: Generation<TGenome>) -> Vec<CompareRecord<TGenome>> {
        let fitness_calc = &self.fitness_calc;

        generation
            .into_iter()
            .filter_map(|predict| fitness_calc.create_compare_record(predict).ok())
            .collect::<Vec<_>>()
    }

    /// Creates a new generation of genomes.
    ///
    /// # Arguments
    ///
    /// - `generation`: The parent generation to breed.
    ///
    /// # Returns
    ///
    /// A new generation of genomes.
    fn new_generation(&self, generation: &[CompareRecord<TGenome>]) -> Vec<CompareRecord<TGenome>> {
        let gen_size = generation.len();
        let mut next_generation = Vec::with_capacity(gen_size);
        let tournament = Tournament::new(self.tournament_size);

        while next_generation.len() < gen_size {
            let left = tournament.select(generation);
            let right = tournament.select(generation);
            let (Some(left), Some(right)) = (left, right) else {
                continue;
            };

            let child = self.breeder.breed(&left.predict, &right.predict);
            if let Ok(fitness) = self.fitness_calc.check(&child) {
                next_generation.push(CompareRecord {
                    fitness,
                    predict: child,
                });
            }
        }
        next_generation
    }
}

/// Builds a new genetic algorithm.
///
/// # Examples
///
/// ```
/// use rand::Rng;
/// use evo::{Breed, Compare, CompareRecord, FitnessCalc, Predict, EvoAlgorithm, TrainingRecord};
///
/// #[derive(Debug, PartialEq, PartialOrd, Clone)]
/// struct Genome {
///     value: f64,
/// }
///
/// impl Predict for Genome {
///     fn predict(&self, input: &[f64]) -> Vec<f64> {
///         input.iter().map(|x| x * self.value).collect()
///     }
/// }
///
/// struct Comparator;
///
/// impl Compare<Genome> for Comparator {
///     fn compare(
///         &self,
///         left: &CompareRecord<&Genome>,
///         right: &CompareRecord<&Genome>,
///     ) -> std::cmp::Ordering {
///         left.fitness.partial_cmp(&right.fitness).unwrap()
///     }
/// }
///
/// struct Breeder {
///     mut_range: f64,
/// }
///
/// impl Breed<Genome> for Breeder {
///     fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
///         Genome {
///             value: (pair.0.value + pair.1.value) / 2.0,
///         }
///     }
///
///     fn mutate(&self, mut genome: Genome) -> Genome {
///         genome.value += rand::thread_rng().gen_range(-1.0..=1.0) * self.mut_range;
///         genome
///     }
/// }
///
/// let fitness_calc = FitnessCalc::builder()
///     .add_training_record(TrainingRecord {
///         input: vec![0.0, 0.0],
///         output: vec![0.0],
///     })
///     .build();
///
/// let breeder = Breeder { mut_range: 0.1 };
/// let comparator = Comparator;
/// let algo = EvoAlgorithm::builder()
///     .breeder(breeder)
///     .fitness_calc(fitness_calc)
///     .build()
///     .unwrap();
/// ```
pub struct Builder<TGenome, TBreeder>
where
    TGenome: Predict + PartialOrd,
    TBreeder: Breed<TGenome>,
{
    elitism: usize,
    tournament_size: usize,
    breeder: Option<BreedManager<TGenome, TBreeder>>,
    fitness_calc: Option<FitnessCalc>,
}

impl<TGenome, TBreeder> Default for Builder<TGenome, TBreeder>
where
    TBreeder: Breed<TGenome>,
    TGenome: Predict + PartialOrd,
{
    fn default() -> Self {
        Self {
            elitism: 1,
            tournament_size: 10,
            breeder: None,
            fitness_calc: None,
        }
    }
}

/// Errors that can occur when building a genetic algorithm.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("breeder not set")]
    BreederNotSet,

    #[error("fitness calc not set")]
    FitnessCalcNotSet,
}

impl<TGenome, TBreeder> Builder<TGenome, TBreeder>
where
    TGenome: Predict + PartialOrd,
    TBreeder: Breed<TGenome>,
{
    /// Builds the genetic algorithm.
    ///
    /// # Returns
    ///
    /// The genetic algorithm.
    ///
    /// # Errors
    ///
    /// - [`Error::BreederNotSet`] if the breeder is not set.
    /// - [`Error::FitnessCalcNotSet`] if the fitness calculator is not set.
    pub fn build(self) -> Result<Run<TGenome, TBreeder>, Error> {
        Ok(Run {
            breeder: self.breeder.ok_or(Error::BreederNotSet)?,
            fitness_calc: self.fitness_calc.ok_or(Error::FitnessCalcNotSet)?,
            elitism: self.elitism,
            tournament_size: self.tournament_size,
        })
    }

    /// Sets the elitism.
    ///
    /// # Arguments
    ///
    /// - `elitism`: The number of elite genomes to keep.
    ///
    /// # Returns
    ///
    /// The builder.
    #[must_use]
    pub fn elitism(mut self, elitism: usize) -> Self {
        self.elitism = elitism;
        self
    }

    /// Sets the tournament size.
    ///
    /// # Arguments
    ///
    /// - `tournament_size`: The number of genomes to select for the tournament.
    ///
    /// # Returns
    ///
    /// The builder.
    #[must_use]
    pub fn tournament_size(mut self, tournament_size: usize) -> Self {
        self.tournament_size = tournament_size;
        self
    }

    /// Sets the breeder.
    ///
    /// # Arguments
    ///
    /// - `breeder`: The breeder to use.
    ///
    /// # Returns
    ///
    /// The builder.
    #[must_use]
    pub fn breeder(mut self, breeder: TBreeder) -> Self {
        self.breeder = Some(BreedManager::new(breeder));
        self
    }

    /// Sets the breeder manager.
    ///
    /// # Arguments
    ///
    /// - `breeder`: The breeder manager to use.
    ///
    /// # Returns
    ///
    /// The builder.
    #[must_use]
    pub fn breeder_manager(mut self, breeder: BreedManager<TGenome, TBreeder>) -> Self {
        self.breeder = Some(breeder);
        self
    }

    /// Sets the fitness calculator.
    ///
    /// # Arguments
    ///
    /// - `fitness_calc`: The fitness calculator to use.
    ///
    /// # Returns
    ///
    /// The builder.
    #[must_use]
    pub fn fitness_calc(mut self, fitness_calc: FitnessCalc) -> Self {
        self.fitness_calc = Some(fitness_calc);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Breed, Compare, TrainingRecord};

    #[test]
    fn test_run() {
        #[derive(Debug, PartialEq, PartialOrd, Clone)]
        struct Genome {
            value: f64,
        }

        impl Predict for Genome {
            fn predict(&self, input: &[f64]) -> Vec<f64> {
                input.iter().map(|x| x * self.value).collect()
            }
        }

        struct Comparator;

        impl Compare<Genome> for Comparator {
            fn compare(
                &self,
                left: &CompareRecord<&Genome>,
                right: &CompareRecord<&Genome>,
            ) -> std::cmp::Ordering {
                left.fitness.partial_cmp(&right.fitness).unwrap()
            }
        }

        struct Breeder;
        impl Breed<Genome> for Breeder {
            fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
                Genome {
                    value: (pair.0.value + pair.1.value) / 2.0,
                }
            }
        }

        let fitness_calc = FitnessCalc::builder()
            .add_training_record(TrainingRecord {
                input: vec![0.0, 0.0],
                output: vec![0.0],
            })
            .build();

        let run = Run {
            breeder: Breeder.to_manager(),
            tournament_size: 2,
            fitness_calc,
            elitism: 1,
        };

        let generation = vec![
            Genome { value: 1.0 },
            Genome { value: 2.0 },
            Genome { value: 3.0 },
        ];

        let actual = run.run(generation.clone());

        assert_eq!(actual.len(), generation.len());
        assert_ne!(actual, generation);
    }
}
