use crate::{CompareRecord, Predict};

pub fn sort_generation<TGenome>(
    candidates: Vec<CompareRecord<TGenome>>,
) -> Vec<CompareRecord<TGenome>>
where
    TGenome: Predict + PartialOrd,
{
    let mut candidates = candidates;
    candidates.sort_by(|left, right| {
        PartialOrd::partial_cmp(left, right).unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates
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
    fn test_sort_generation() {
        impl std::fmt::Debug for CompareRecord<Predictor> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("CompareRecord")
                    .field("fitness", &self.fitness)
                    .field("predict", &self.predict.value)
                    .finish()
            }
        }

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
        let expected = vec![
            CompareRecord {
                fitness: 1.0,
                predict: Predictor { value: 1.0 },
            },
            CompareRecord {
                fitness: 2.0,
                predict: Predictor { value: 2.0 },
            },
            CompareRecord {
                fitness: 3.0,
                predict: Predictor { value: 3.0 },
            },
        ];
        let actual = sort_generation(candidates);
        assert_eq!(actual, expected);
    }
}
