use crate::{CompareRecord, Predict};
use rand::prelude::*;

/// Tournament selection algorithm.
pub struct Tournament {
    tournament_size: usize,
}

impl Tournament {
    /// Create a new tournament selection algorithm.
    ///
    /// # Arguments
    ///
    /// * `tournament_size` - The number of candidates to select from.
    ///
    /// # Returns
    ///
    /// The tournament selection algorithm.
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }

    /// Select a candidate from a list of candidates.
    ///
    /// # Arguments
    ///
    /// * `candidates` - The list of candidates to select from.
    ///
    /// # Returns
    ///
    /// The selected candidate.
    pub fn select<'x, TGenome>(
        &self,
        candidates: &'x [CompareRecord<TGenome>],
    ) -> Option<&'x CompareRecord<TGenome>>
    where
        TGenome: Predict + PartialOrd,
    {
        let mut winner = None;
        for candidate in self.tournament_iter(candidates) {
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

    /// Create an iterator over the tournament candidates.
    ///
    /// # Arguments
    ///
    /// * `candidates` - The list of candidates to select from.
    ///
    /// # Returns
    ///
    /// An iterator over the tournament candidates.
    fn tournament_iter<'x, TGenome>(
        &self,
        candidates: &'x [CompareRecord<TGenome>],
    ) -> impl Iterator<Item = &'x CompareRecord<TGenome>>
    where
        TGenome: Predict + PartialOrd,
    {
        let tournament_size = self.tournament_size(candidates);

        let mut indexes = (0..candidates.len()).collect::<Vec<_>>();
        indexes.shuffle(&mut thread_rng());
        indexes
            .into_iter()
            .take(tournament_size)
            .map(move |id| &candidates[id])
    }

    /// Get the tournament size.
    ///
    /// # Arguments
    ///
    /// * `candidates` - The list of candidates to select from.
    ///
    /// # Returns
    ///
    /// The tournament size.
    pub fn tournament_size<TGenome>(&self, candidates: &[TGenome]) -> usize {
        usize::min(self.tournament_size, candidates.len())
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

    #[test]
    fn test_tournament_size() {
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

        let result = tournament.tournament_size(&candidates);

        assert_eq!(result, 2);
    }
}
