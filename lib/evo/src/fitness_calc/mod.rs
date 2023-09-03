use thiserror::Error;

/// A fitness calculator record.
pub struct TrainingRecord {
    /// The input data for this training record.
    pub input: Vec<f64>,

    /// The expected output data for this training record.
    pub output: Vec<f64>,
}

impl TrainingRecord {
    /// Calculate the mean squared error between the actual values provided
    /// and the expected outputs. Values closer to 0.0 are better.
    ///
    /// # Arguments
    ///
    /// - `actual` is the actual output.
    ///
    /// # Returns
    ///
    /// An iterator of mean squared errors.
    fn get_mse<'a>(&'a self, actual: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
        Iterator::zip(self.output.iter(), actual.iter())
            .map(|(expected, actual)| (expected - actual).powi(2))
    }
}

/// A trait for prediction functions.
///
/// # Examples
///
/// ```
/// use evo::Predict;
///
/// struct Predictor;
///
/// impl Predict for Predictor {
///    fn predict(&self, input: &[f64]) -> Vec<f64> {
///       input.iter().map(|x| x * 2.0).collect()
///    }
/// }
///
/// let predictor = Predictor;
/// let actual = predictor.predict(&[1.0, 2.0, 3.0]);
/// let expected = vec![2.0, 4.0, 6.0];
///
/// assert_eq!(actual, expected);
/// ```
pub trait Predict {
    /// Predict the output for the given input.
    ///
    /// # Arguments
    ///
    /// - `input` is the input data.
    ///
    /// # Returns
    ///
    /// The predicted output.
    fn predict(&self, input: &[f64]) -> Vec<f64>;
}

/// A fitness calculator for the evolutionary algorithm.
///
/// # Examples
///
/// ```
/// use evo::FitnessCalc;
///
/// let fitness_calc = FitnessCalc::builder().build();
/// ```
pub struct FitnessCalc {
    training_data: Vec<TrainingRecord>,
    band_width: f64,
}

/// An error that can occur when calculating fitness.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("cannot convert")]
    CannotConvert,

    #[error("result is NaN")]
    ResultNaN,

    #[error("result is infinite")]
    ResultInfinite,
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
fn convert(x: usize) -> Result<f64, Error> {
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
fn checked_divide(numerator: f64, denominator: f64) -> Result<f64, Error> {
    let result = numerator / denominator;
    if result.is_nan() {
        Err(Error::ResultNaN)
    } else if result.is_infinite() {
        Err(Error::ResultInfinite)
    } else {
        Ok(result)
    }
}

impl FitnessCalc {
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
    fn get_mse_iter<'n>(
        &'n self,
        predict: &'n impl Predict,
    ) -> impl Iterator<Item = Vec<f64>> + 'n {
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
    pub fn check(&self, predict: &impl Predict) -> Result<f64, Error> {
        let len = convert(self.training_data.len())?;
        let mse_sum = self
            .get_mse_iter(predict)
            .map(|x| {
                let x_len = convert(x.len())?;
                let x_sum = x.iter().sum::<f64>();
                checked_divide(x_sum, x_len)
            })
            .sum::<Result<f64, Error>>()?;

        checked_divide(mse_sum, len)
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
    band_width: f64,
    training_data: Vec<TrainingRecord>,
}

impl Builder {
    /// Set the band width for the fitness calc.
    ///
    /// # Arguments
    ///
    /// - `threshold` is the band width.
    ///
    /// # Returns
    ///
    /// The builder.
    #[must_use]
    pub fn band_width(mut self, threshold: f64) -> Self {
        self.band_width = threshold;
        self
    }

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
    pub fn build(self) -> FitnessCalc {
        FitnessCalc {
            training_data: self.training_data,
            band_width: self.band_width,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPredict;

    impl Predict for TestPredict {
        fn predict(&self, _input: &[f64]) -> Vec<f64> {
            vec![0.0]
        }
    }

    #[test]
    fn test_fitness_calc() {
        let fitness_calc = FitnessCalc::builder()
            .add_training_record(TrainingRecord {
                input: vec![0.0, 0.0],
                output: vec![0.0],
            })
            .build();
        let fitness = fitness_calc.check(&TestPredict);

        assert_eq!(fitness, Ok(0.0));
    }
}
