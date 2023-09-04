use super::{inject_genomes, sort_generation, unrank_generation, Tournament};
use crate::{Breed, BreedManager, Compare, CompareRecord, FitnessCalc, Generation, Predict};
use rand::Rng;

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
    pub fn run(&self, generation: Generation<TGenome>) -> Generation<TGenome> {
        let ranked_generation = self.rank_generation(generation);

        let next_generation = self.new_generation(&ranked_generation);
        let next_generation = unrank_generation(next_generation);

        let sorted_generation = sort_generation(ranked_generation);
        let sorted_generation = unrank_generation(sorted_generation);

        let elitism = std::cmp::min(self.elitism, sorted_generation.len());
        let mut elite = sorted_generation;
        elite.truncate(elitism);
        let elite = elite;

        inject_genomes(next_generation, elite)
    }

    fn rank_generation(&self, generation: Generation<TGenome>) -> Vec<CompareRecord<TGenome>> {
        let fitness_calc = &self.fitness_calc;

        generation
            .into_iter()
            .filter_map(|predict| fitness_calc.create_compare_record(predict).ok())
            .collect::<Vec<_>>()
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Breed, TrainingRecord};

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
