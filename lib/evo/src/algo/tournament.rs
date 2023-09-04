use crate::{CompareRecord, Predict};
use rand::Rng;

pub struct Tournament<TGenome> {
    _phantom: std::marker::PhantomData<TGenome>,
    tournament_size: usize,
}

impl<TGenome> Tournament<TGenome>
where
    TGenome: Predict + PartialOrd,
{
    pub fn new(tournament_size: usize) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
            tournament_size,
        }
    }

    pub fn select<'x>(
        &self,
        candidates: &'x [CompareRecord<TGenome>],
    ) -> Option<&'x CompareRecord<TGenome>> {
        let candidate_count = candidates.len();
        let tournament_size = usize::min(candidate_count, self.tournament_size);

        let mut winner = None;
        for _ in 0..tournament_size {
            let id = rand::thread_rng().gen_range(0..candidate_count);
            let candidate = candidates.get(id)?;

            winner = Some(match winner {
                None => candidate,
                Some(winner) => match PartialOrd::partial_cmp(winner, candidate) {
                    Some(std::cmp::Ordering::Less) => winner,
                    _ => candidate,
                },
            });
        }

        winner
    }
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

    impl std::fmt::Debug for CompareRecord<Predictor> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CompareRecord")
                .field("fitness", &self.fitness)
                .field("predict", &self.predict.value)
                .finish()
        }
    }

    #[test]
    fn test_tournament_selection() {
        let candidates = vec![
            CompareRecord {
                fitness: 1.0,
                predict: Predictor { value: 1.0 },
            },
            CompareRecord {
                fitness: 3.0,
                predict: Predictor { value: 3.0 },
            },
            CompareRecord {
                fitness: 2.0,
                predict: Predictor { value: 2.0 },
            },
        ];

        let tournament = Tournament::new(candidates.len());

        let result = tournament.select(&candidates);

        assert_eq!(
            result,
            Some(&CompareRecord {
                fitness: 1.0,
                predict: Predictor { value: 1.0 },
            })
        );
    }

    #[test]
    fn test_tournament_selection_with_tournament_size() {
        let candidates = vec![
            CompareRecord {
                fitness: 1.0,
                predict: Predictor { value: 1.0 },
            },
            CompareRecord {
                fitness: 3.0,
                predict: Predictor { value: 3.0 },
            },
            CompareRecord {
                fitness: 2.0,
                predict: Predictor { value: 2.0 },
            },
        ];

        let tournament = Tournament::new(2);

        let result = tournament.select(&candidates);

        assert_ne!(
            result,
            Some(&CompareRecord {
                fitness: 3.0,
                predict: Predictor { value: 3.0 },
            })
        );
    }
}
