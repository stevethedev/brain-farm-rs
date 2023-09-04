use crate::{Generation, Predict};
use rand::Rng;

pub fn inject_genomes<TGenome>(
    mut generation: Generation<TGenome>,
    elite: Generation<TGenome>,
) -> Generation<TGenome>
where
    TGenome: Predict + PartialOrd,
{
    let generation_size = generation.len();
    for genome in elite {
        let generation_index = rand::thread_rng().gen_range(0..generation_size);
        generation[generation_index] = genome;
    }
    generation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Predictor {
        value: f64,
    }

    impl Predict for Predictor {
        fn predict(&self, input: &[f64]) -> Vec<f64> {
            input.iter().map(|x| x * self.value).collect()
        }
    }

    impl PartialOrd for Predictor {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.value.partial_cmp(&other.value)
        }
    }

    #[test]
    fn test_inject_genomes() {
        let generation = vec![
            Predictor { value: 1.0 },
            Predictor { value: 2.0 },
            Predictor { value: 3.0 },
        ];
        let elite = vec![Predictor { value: 4.0 }];
        let expected = vec![
            Predictor { value: 4.0 },
            Predictor { value: 2.0 },
            Predictor { value: 3.0 },
        ];

        let result = inject_genomes(generation, elite);

        assert_ne!(
            result.into_iter().find(|p| p == &Predictor { value: 4.0 }),
            None
        );
    }
}
