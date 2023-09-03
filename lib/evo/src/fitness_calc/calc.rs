use super::{Error, Predict, Result, TrainingRecord};
use crate::{Compare, CompareRecord};

/// A fitness calculator for the evolutionary algorithm.
///
/// # Examples
///
/// ```
/// use evo::FitnessCalc;
///
/// let fitness_calc = FitnessCalc::builder().build();
/// ```
pub struct Calc {
    training_data: Vec<TrainingRecord>,
}

/// Convert a `usize` to a `f64`.
///
/// # Arguments
///
/// - `x` is the `usize` to convert.
///
/// # Returns
///
/// The `f64` value.
///
/// # Errors
///
/// If the `usize` cannot be converted to a `f64`.
fn convert(x: usize) -> Result<f64> {
    #[allow(clippy::cast_precision_loss)]
    let result = x as f64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    if result as usize == x {
        Ok(result)
    } else {
        Err(Error::CannotConvert)
    }
}

/// Divide two `f64` values, checking for `NaN` and `Infinite` results.
///
/// # Arguments
///
/// - `numerator` is the numerator.
/// - `denominator` is the denominator.
///
/// # Returns
///
/// The result of the division.
///
/// # Errors
///
/// If the result is `NaN` or `Infinite`.
fn checked_divide(numerator: f64, denominator: f64) -> Result<f64> {
    let result = numerator / denominator;
    if result.is_nan() {
        Err(Error::ResultNaN)
    } else if result.is_infinite() {
        Err(Error::ResultInfinite)
    } else {
        Ok(result)
    }
}

impl Calc {
    /// Create a new builder.
    ///
    /// # Returns
    ///
    /// A new builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::FitnessCalc;
    ///
    /// let fitness_calc = FitnessCalc::builder().build();
    /// ```
    #[must_use]
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Get the mean squared error for each training record.
    ///
    /// # Arguments
    ///
    /// - `predict` is the prediction function.
    ///
    /// # Returns
    ///
    /// An iterator of mean squared errors.
    fn get_mse_iter<'n, P>(&'n self, predict: &'n P) -> impl Iterator<Item = Vec<f64>> + 'n
    where
        P: Predict,
    {
        self.training_data.iter().map(move |t_record| {
            let actual = predict.predict(&t_record.input);
            t_record.get_mse(&actual).collect()
        })
    }

    /// Use the prediction function to check the fitness of an entity.
    ///
    /// # Arguments
    ///
    /// - `predict` is the prediction function.
    ///
    /// # Returns
    ///
    /// The fitness of the entity.
    ///
    /// # Errors
    ///
    /// If the number of training records cannot be converted to a `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::{FitnessCalc, Predict, TrainingRecord};
    ///
    /// struct Predictor;
    ///
    /// impl Predict for Predictor {
    ///     fn predict(&self, _input: &[f64]) -> Vec<f64> {
    ///         vec![0.0]
    ///     }
    /// }
    ///
    /// let fitness_calc = FitnessCalc::builder()
    ///     .add_training_record(TrainingRecord {
    ///         input: vec![1.0, 2.0, 3.0, 4.0],
    ///         output: vec![1.0, 2.0, 3.0, 4.0],
    ///     })
    ///     .build();
    /// let fitness = fitness_calc.check(&Predictor).unwrap();
    ///
    /// assert_eq!(fitness, 1.0);
    /// ```
    pub fn check<P>(&self, predict: &P) -> Result<f64>
    where
        P: Predict,
    {
        let len = convert(self.training_data.len())?;
        let mse_sum = self
            .get_mse_iter(predict)
            .map(|x| {
                let x_len = convert(x.len())?;
                let x_sum = x.iter().sum::<f64>();
                checked_divide(x_sum, x_len)
            })
            .sum::<Result<f64>>()?;

        checked_divide(mse_sum, len)
    }

    /// Get the best entity from a set of entities, where the best entity is the one with the lowest fitness value.
    /// If two entities have the same fitness value, the first one is returned.
    /// If no entities are provided, `None` is returned.
    ///
    /// # Arguments
    ///
    /// - `entities` is the set of entities to check.
    /// - `compare` is the comparison function.
    ///
    /// # Returns
    ///
    /// The best entity, or `None` if no entities are provided.
    ///
    /// # Errors
    ///
    /// If the fitness of any entity cannot be calculated.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::{Compare, CompareRecord, FitnessCalc, Predict, TrainingRecord};
    /// use std::cmp::{Ordering, Eq, PartialEq};
    ///
    /// #[derive(Debug)]
    /// struct Predictor;
    ///
    /// impl Predict for Predictor {
    ///     fn predict(&self, _input: &[f64]) -> Vec<f64> {
    ///         vec![0.0]
    ///     }
    /// }
    ///
    /// impl PartialEq for Predictor {
    ///     fn eq(&self, _other: &Self) -> bool {
    ///         true
    ///     }
    /// }
    ///
    /// impl Eq for Predictor {}
    ///
    /// impl PartialOrd for Predictor {
    ///     fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
    ///         Some(Ordering::Equal)
    ///     }
    /// }
    ///
    /// struct Comparator;
    ///
    /// impl Compare<Predictor> for Comparator {
    ///     fn compare(&self, left: &CompareRecord<Predictor>, right: &CompareRecord<Predictor>) -> Ordering {
    ///         match (left.fitness.is_nan() || left.fitness.is_infinite(), right.fitness.is_nan() || right.fitness.is_infinite()) {
    ///             (true, true) => Ordering::Equal,
    ///             (true, false) => Ordering::Greater,
    ///             (false, true) => Ordering::Less,
    ///             (false, false) => PartialOrd::partial_cmp(&left.fitness, &right.fitness).unwrap_or(Ordering::Equal),
    ///         }
    ///     }
    /// }
    ///
    /// let fitness_calc = FitnessCalc::builder()
    ///     .add_training_record(TrainingRecord {
    ///         input: vec![1.0, 2.0, 3.0, 4.0],
    ///         output: vec![1.0, 2.0, 3.0, 4.0],
    ///     })
    ///     .build();
    ///
    /// let best = fitness_calc.best_entity(&[Predictor], &Comparator);
    ///
    /// assert_eq!(best, Ok(Some(&Predictor)));
    /// ```
    pub fn best_entity<'x, P, C>(&self, entities: &'x [P], compare: &C) -> Result<Option<&'x P>>
    where
        P: Predict + PartialOrd,
        C: Compare<P>,
    {
        let vector = entities
            .iter()
            .map(|predict| {
                let fitness = self.check(predict)?;
                Ok(CompareRecord { fitness, predict })
            })
            .collect::<Result<Vec<_>>>()?;

        let result = vector
            .into_iter()
            .min_by(|left, right| compare.compare(left, right))
            .map(|record| record.predict);

        Ok(result)
    }
}

/// A builder for `FitnessCalc`s.
///
/// # Examples
///
/// ```
/// use evo::FitnessCalc;
///
/// let fitness_calc = FitnessCalc::builder().build();
/// ```
#[derive(Default)]
pub struct Builder {
    training_data: Vec<TrainingRecord>,
}

impl Builder {
    /// Add training data to the fitness calc.
    ///
    /// # Arguments
    ///
    /// - `input` is the input data.
    /// - `output` is the expected output data.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::{FitnessCalc, TrainingRecord};
    ///
    /// let fitness_calc = FitnessCalc::builder()
    ///     .add_training_record(TrainingRecord { input: vec![0.0, 0.0], output: vec![0.0] })
    ///     .build();
    /// ```
    #[must_use]
    pub fn add_training_record(mut self, record: TrainingRecord) -> Self {
        self.training_data.push(record);
        self
    }

    /// Build the fitness calc.
    ///
    /// # Returns
    ///
    /// The fitness calc.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::FitnessCalc;
    ///
    /// let fitness_calc = FitnessCalc::builder().build();
    /// ```
    pub fn build(self) -> Calc {
        Calc {
            training_data: self.training_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn test_fitness_calc() {
        let fitness_calc = Calc::builder()
            .add_training_record(TrainingRecord {
                input: vec![0.0, 0.0],
                output: vec![0.0],
            })
            .build();

        let actual = fitness_calc.check(&TestPredict).unwrap();

        assert!(actual.abs() <= f64::EPSILON, "expected ~0.0, got {actual}");
    }

    #[test]
    fn test_fitness_calc_best_entity() {
        let fitness_calc = Calc::builder()
            .add_training_record(TrainingRecord {
                input: vec![0.0, 0.0],
                output: vec![0.0],
            })
            .build();

        let actual = fitness_calc.best_entity(&[Predictor(1.0), Predictor(2.0)], &Comparator);

        assert_eq!(actual, Ok(Some(&Predictor(1.0))));
    }

    #[derive(Ord, PartialOrd, Eq, PartialEq, Debug)]
    struct TestPredict;

    impl Predict for TestPredict {
        fn predict(&self, _input: &[f64]) -> Vec<f64> {
            vec![0.0]
        }
    }

    #[derive(PartialOrd, PartialEq, Debug)]
    struct Predictor(f64);

    impl Eq for Predictor {}

    impl Predict for Predictor {
        fn predict(&self, input: &[f64]) -> Vec<f64> {
            input.iter().map(|x| x * self.0).collect()
        }
    }

    struct Comparator;

    impl Compare<Predictor> for Comparator {
        fn compare(
            &self,
            left: &CompareRecord<Predictor>,
            right: &CompareRecord<Predictor>,
        ) -> Ordering {
            left.fitness.partial_cmp(&right.fitness).unwrap()
        }
    }
}
