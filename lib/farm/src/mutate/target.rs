use rand::{thread_rng, Rng};

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

/// Vector mutation instructions.
pub enum VecMutation<T> {
    /// Insert an element at the given index.
    Insert(usize, T),

    /// Replace an element at the given index.
    Replace(usize, T),

    /// Remove an element at the given index.
    Remove(usize),

    /// Swap two elements at the given indices.
    Swap(usize, usize),

    /// Reverse the elements between the given indices.
    Reverse(usize, usize),
}

impl<T> VecMutation<T> {
    /// Generate a random mutation.
    ///
    /// # Arguments
    ///
    /// - `len` - The length of the vector.
    ///
    /// # Returns
    ///
    /// The random mutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::mutate::VecMutation;
    ///
    /// let mutation = VecMutation::new(10, || 0);
    /// ```
    pub fn new(len: usize, factory: impl Fn() -> T) -> Self {
        let mut rng = thread_rng();
        match rng.gen_range(0..5) {
            0 => Self::Insert(rng.gen_range(0..len), factory()),
            1 => Self::Replace(rng.gen_range(0..len), factory()),
            2 => Self::Remove(rng.gen_range(0..len)),
            3 => Self::Swap(rng.gen_range(0..len), rng.gen_range(0..len)),
            _ => Self::Reverse(rng.gen_range(0..len), rng.gen_range(0..len)),
        }
    }

    /// Apply the mutation to a vector.
    ///
    /// # Arguments
    ///
    /// - `vec` - The vector to mutate.
    /// - `factory` - A factory function to create new elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::mutate::VecMutation;
    ///
    /// let mut vec = vec![0, 1, 2, 3, 4, 5];
    /// let mutation = VecMutation::new(vec.len(), || 0);
    /// mutation.apply(&mut vec);
    /// ```
    pub fn apply(self, vec: &mut Vec<T>) {
        match self {
            Self::Insert(index, element) => vec.insert(index, element),
            Self::Replace(index, element) => vec[index] = element,
            Self::Remove(index) => {
                vec.remove(index);
            }
            Self::Swap(i_index, j_index) => vec.swap(i_index, j_index),
            Self::Reverse(i_index, j_index) => {
                let min = usize::min(i_index, j_index);
                let max = usize::max(i_index, j_index);

                if min != max {
                    vec[min..=max].reverse();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_mutation_insert() {
        let mut vec = vec![0, 1, 2, 3, 4, 5];

        let mutation = VecMutation::Insert(0, 10);
        mutation.apply(&mut vec);
        assert_eq!(vec, vec![10, 0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_vec_mutation_replace() {
        let mut vec = vec![0, 1, 2, 3, 4, 5];

        let mutation = VecMutation::Replace(3, 11);
        mutation.apply(&mut vec);
        assert_eq!(vec, vec![0, 1, 2, 11, 4, 5]);
    }

    #[test]
    fn test_vec_mutation_remove() {
        let mut vec = vec![0, 1, 2, 3, 4, 5];

        let mutation = VecMutation::Remove(3);
        mutation.apply(&mut vec);
        assert_eq!(vec, vec![0, 1, 2, 4, 5]);
    }

    #[test]
    fn test_vec_mutation_swap() {
        let mut vec = vec![0, 1, 2, 3, 4, 5];

        let mutation = VecMutation::Swap(0, 1);
        mutation.apply(&mut vec);
        assert_eq!(vec, vec![1, 0, 2, 3, 4, 5]);
    }

    #[test]
    fn test_vec_mutation_reverse() {
        let mut vec = vec![0, 1, 2, 3, 4, 5];

        let mutation = VecMutation::Reverse(1, 4);
        mutation.apply(&mut vec);
        assert_eq!(vec, vec![0, 4, 3, 2, 1, 5]);
    }
}
