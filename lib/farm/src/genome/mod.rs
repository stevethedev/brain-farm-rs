pub mod activator;
pub mod layer;
pub mod network;
pub mod neuron;

use rand::{random, thread_rng, Rng};

/// Enable automatic generation of a gene or genome.
///
/// # Examples
///
/// ```
/// use rand::{Rng, thread_rng};
/// use farm::genome::Generate;
///
/// struct Genome {
///     value: f64,
/// }
///
/// impl Generate<std::ops::Range<f64>> for Genome {
///     fn generate(config: std::ops::Range<f64>) -> Self {
///         let value = thread_rng().gen_range(config.start..config.end);
///         Self { value }
///     }
/// }
///
/// let config = 0.0..1.0;
/// let genome = Genome::generate(config);
///
/// assert!(
///     genome.value >= 0.0 && genome.value < 1.0,
///     "expected {value} to be between {start} and {end}",
///     value = genome.value,
///     start = 0.0,
///     end = 1.0,
/// );
/// ```
pub trait Generate<TConfig> {
    fn generate(config: TConfig) -> Self;
}

impl Generate<std::ops::Range<f64>> for f64 {
    fn generate(config: std::ops::Range<f64>) -> Self {
        thread_rng().gen_range(config.start..config.end)
    }
}

impl Generate<std::ops::RangeInclusive<f64>> for f64 {
    fn generate(config: std::ops::RangeInclusive<f64>) -> Self {
        thread_rng().gen_range(*config.start()..=*config.end())
    }
}

/// Use a genome to create a new entity.
///
/// # Examples
///
/// ```
/// use farm::genome::Create;
///
/// struct Entity {
///    value: f64,
/// }
///
/// impl Create<Entity> for f64 {
///     fn create(&self) -> Entity {
///         Entity { value: *self }
///     }
/// }
///
/// let entity = 1.0.create();
///
/// assert_eq!(entity.value, 1.0);
/// ```
pub trait Create<TEntity> {
    fn create(&self) -> TEntity;
}

/// Extract the genome from an entity.
///
/// # Examples
///
/// ```
/// use farm::genome::Extract;
///
/// struct Entity {
///     value: f64,
/// }
///
/// impl Extract<f64> for Entity {
///     fn genome(&self) -> f64 {
///         self.value
///     }
/// }
///
/// let entity = Entity { value: 1.0 };
/// let genome = entity.genome();
///
/// assert_eq!(genome, 1.0);
/// ```
pub trait Extract<TEntity> {
    fn genome(&self) -> TEntity;
}

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
    #[must_use]
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

        if f64::abs(max - min) < f64::EPSILON {
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
            other.iter().skip(min_size)
        } else {
            self.iter().skip(min_size)
        };

        Iterator::zip(self.iter(), other.iter())
            .map(|(a, b)| a.crossover(b))
            .chain(rest.map(Clone::clone))
            .collect()
    }
}
