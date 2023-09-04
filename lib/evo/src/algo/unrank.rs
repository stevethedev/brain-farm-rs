use crate::{CompareRecord, Generation, Predict};

pub fn unrank_generation<TGenome>(
    ranked_generation: Vec<CompareRecord<TGenome>>,
) -> Generation<TGenome>
where
    TGenome: Predict + PartialOrd,
{
    ranked_generation
        .into_iter()
        .map(|x| x.predict)
        .collect::<Generation<TGenome>>()
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
    fn test_unrank_generation() {
        let ranked_generation = vec![
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
            Predictor { value: 1.0 },
            Predictor { value: 3.0 },
            Predictor { value: 2.0 },
        ];

        let result = unrank_generation(ranked_generation);

        assert_eq!(result, expected);
    }
}
