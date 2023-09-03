/// A fitness calculator record.
pub struct Record {
    /// The input data for this training record.
    pub input: Vec<f64>,

    /// The expected output data for this training record.
    pub output: Vec<f64>,
}

impl Record {
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
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::TrainingRecord;
    ///
    /// let record = TrainingRecord {
    ///     input: vec![1.0, 2.0, 3.0],
    ///     output: vec![2.0, 4.0, 6.0],
    /// };
    ///
    /// let actual = vec![2.0, 4.0, 6.0];
    ///
    /// let mse = record.get_mse(&actual).collect::<Vec<f64>>();
    ///
    /// assert_eq!(mse, vec![0.0, 0.0, 0.0]);
    /// ```
    pub fn get_mse<'a>(&'a self, actual: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
        Iterator::zip(self.output.iter(), actual.iter())
            .map(|(expected, actual)| (expected - actual).powi(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record() {
        let record = Record {
            input: vec![1.0, 2.0, 3.0],
            output: vec![2.0, 4.0, 6.0],
        };

        let actual = vec![2.0, 4.0, 6.0];

        let mse = record.get_mse(&actual).collect::<Vec<f64>>();

        assert_eq!(mse, vec![0.0, 0.0, 0.0]);
    }
}
