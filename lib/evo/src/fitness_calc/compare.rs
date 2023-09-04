use super::Predict;

/// A trait for comparing entities.
///
/// # Examples
///
/// ```
/// use evo::{Compare, CompareRecord, Predict};
/// use std::cmp::Ordering;
///
/// struct Predictor;
///
/// impl Predict for Predictor {
///    fn predict(&self, _input: &[f64]) -> Vec<f64> {
///       vec![0.0]
///    }
/// }
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
/// impl Ord for Predictor {
///     fn cmp(&self, _other: &Self) -> Ordering {
///         Ordering::Equal
///     }
/// }
///
/// let left = CompareRecord {
///    fitness: 0.0,
///    predict: &Predictor,
/// };
///
/// let right = CompareRecord {
///    fitness: 1.0,
///    predict: &Predictor,
/// };
///
/// struct Comparator;
///
/// impl Compare<Predictor> for Comparator {
///     fn compare(&self, left: &CompareRecord<&Predictor>, right: &CompareRecord<&Predictor>) -> Ordering {
///         left.fitness.partial_cmp(&right.fitness).unwrap()
///     }
/// }
///
/// let compare = Comparator;
/// let ordering = compare.compare(&left, &right);
///
/// assert_eq!(ordering, Ordering::Less);
/// ```
pub trait Compare<P>
where
    P: Predict + PartialOrd,
{
    /// Compare two entities.
    fn compare(&self, left: &Record<&P>, right: &Record<&P>) -> std::cmp::Ordering;

    fn compare_raw(&self, left: &Record<P>, right: &Record<P>) -> std::cmp::Ordering {
        let left = Record {
            fitness: left.fitness,
            predict: &left.predict,
        };
        let right = Record {
            fitness: right.fitness,
            predict: &right.predict,
        };
        self.compare(&left, &right)
    }
}

/// A record for comparing entities.
pub struct Record<P>
where
    P: Predict + PartialOrd,
{
    /// The fitness of the entity.
    pub fitness: f64,

    /// The prediction function for the entity.
    pub predict: P,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    struct Predictor;

    impl Predict for Predictor {
        fn predict(&self, _input: &[f64]) -> Vec<f64> {
            vec![0.0]
        }
    }
    impl PartialEq for Predictor {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }

    impl Eq for Predictor {}

    impl PartialOrd for Predictor {
        fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
            Some(Ordering::Equal)
        }
    }

    impl Ord for Predictor {
        fn cmp(&self, _other: &Self) -> Ordering {
            Ordering::Equal
        }
    }

    #[test]
    fn compare() {
        struct Comparator;
        impl Compare<Predictor> for Comparator {
            fn compare(&self, left: &Record<&Predictor>, right: &Record<&Predictor>) -> Ordering {
                left.fitness.partial_cmp(&right.fitness).unwrap()
            }
        }

        let left = Record {
            fitness: 0.0,
            predict: &Predictor,
        };

        let right = Record {
            fitness: 1.0,
            predict: &Predictor,
        };

        let compare = Comparator;
        let ordering = compare.compare(&left, &right);

        assert_eq!(ordering, Ordering::Less);
    }
}
