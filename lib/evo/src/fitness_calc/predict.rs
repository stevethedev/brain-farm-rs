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

impl<P> Predict for &P
where
    P: Predict,
{
    fn predict(&self, input: &[f64]) -> Vec<f64> {
        (*self).predict(input)
    }
}
