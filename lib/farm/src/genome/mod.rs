pub mod activator;
pub mod neuron;

use rand::{random, thread_rng, Rng};

/// Enable crossover for a gene or genome.
pub trait Crossover {
    /// Crossover the target.
    ///
    /// # Arguments
    ///
    /// - `other` - The other target to crossover with.
    ///
    /// # Returns
    ///
    /// The crossovered target.
    fn crossover(&self, other: &Self) -> Self;
}

/// Implement `Target` for `f64`.
///
/// # Examples
///
/// ```
/// use farm::genome::Crossover;
///
/// let left = 0.0;
/// let right = 1.0;
///
/// let target = left.crossover(&right);
/// assert!(target > 0.0 && target < 1.0, "expected {target} to be between {left} and {right}");
/// ```
impl Crossover for f64 {
    fn crossover(&self, other: &Self) -> Self {
        let a = if self.is_nan() || self.is_infinite() {
            thread_rng().gen_range(-1.0..=1.0)
        } else {
            *self
        };
        let b = if other.is_nan() || other.is_infinite() {
            thread_rng().gen_range(-1.0..=1.0)
        } else {
            *other
        };

        let (min, max) = (a.min(b), a.max(b));

        if min == max {
            min
        } else {
            thread_rng().gen_range(min..=max)
        }
    }
}

/// Implement `Target` for `bool`.
///
/// # Examples
///
/// ```
/// use farm::genome::Crossover;
///
/// let left = false;
/// let right = true;
///
/// let target = left.crossover(&right);
/// ```
impl Crossover for bool {
    fn crossover(&self, other: &Self) -> Self {
        if random::<bool>() {
            *self
        } else {
            *other
        }
    }
}

/// Implement `Crossover` for `Vec<Crossover + Clone>`.
///
/// # Examples
///
/// ```
/// use farm::genome::Crossover;
///
/// let left = vec![ vec![0.0], vec![1.0], vec![2.0] ];
/// let right = vec![ vec![3.0], vec![4.0], vec![5.0] ];
///
/// let target = left.crossover(&right);
/// ```
impl<T> Crossover for Vec<T>
where
    T: Crossover + Clone,
{
    fn crossover(&self, other: &Self) -> Self {
        let self_len = self.len();
        let other_len = other.len();

        let min_size = usize::min(self_len, other_len);

        let rest = if self_len < other_len {
            other.into_iter().skip(min_size)
        } else {
            self.into_iter().skip(min_size)
        };

        Iterator::zip(self.iter(), other.iter())
            .map(|(a, b)| a.crossover(b))
            .chain(rest.map(Clone::clone))
            .collect()
    }
}
