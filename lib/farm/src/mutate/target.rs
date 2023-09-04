/// Trait for types that can be mutated.
///
/// # Examples
///
/// ```
/// use farm::mutate::{Target, Mutator};
///
/// let mutator = Mutator::builder().build();
///
/// let mut target = 0.0;
/// target = target.mutate(&mutator);
/// ```
pub trait Target {
    /// Mutate the target.
    ///
    /// # Arguments
    ///
    /// - `mutator` - The mutator to use.
    ///
    /// # Returns
    ///
    /// The mutated target.
    fn mutate(self, mutator: &super::Mutator) -> Self;
}

/// Implement `Target` for `f64`.
///
/// # Examples
///
/// ```
/// use farm::mutate::{Target, Mutator};
///
/// let mutator = Mutator::builder().build();
///
/// let mut target = 0.0;
/// target = target.mutate(&mutator);
/// ```
impl Target for f64 {
    fn mutate(mut self, mutator: &super::Mutator) -> Self {
        if mutator.mutation_size() > 0.0 && mutator.check_mutate() {
            self += rand::random::<f64>() * mutator.mutation_size();
        }

        self
    }
}

/// Implement `Target` for `bool`.
///
/// # Examples
///
/// ```
/// use farm::mutate::{Target, Mutator};
///
/// let mutator = Mutator::builder().build();
///
/// let mut target = false;
/// target = target.mutate(&mutator);
/// ```
impl Target for bool {
    fn mutate(mut self, mutator: &super::Mutator) -> Self {
        if mutator.mutation_size() > 0.0 && mutator.check_mutate() {
            self = !self;
        }

        self
    }
}

/// Implement `Target` for `Vec<impl Target>`
///
/// # Examples
///
/// ```
/// use farm::mutate::{Target, Mutator};
///
/// let mutator = Mutator::builder().build();
///
/// let mut target = vec![0.0, 1.0, 2.0];
/// target = target.mutate(&mutator);
/// ```
impl<T> Target for Vec<T>
where
    T: Target,
{
    fn mutate(self, mutator: &super::Mutator) -> Self {
        self.into_iter().map(|t| t.mutate(mutator)).collect()
    }
}
