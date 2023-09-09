use rand::{thread_rng, Rng};

/// A struct that manages the chances for mutating a genome.
///
/// # Examples
///
/// ```
/// use farm::mutate::Mutator;
///
/// let mutator = Mutator::builder().build();
/// ```
#[derive(Clone, Copy)]
pub struct Mutator {
    /// The chance to mutate a genome.
    mutation_rate: f64,

    /// The degree of mutation.
    mutation_size: f64,
}

impl Mutator {
    /// Create a new mutator builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::mutate::Mutator;
    ///
    /// let mutator = Mutator::builder().build();
    /// ```
    #[must_use]
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Check if a genome should be mutated.
    ///
    /// # Returns
    ///
    /// True if the genome should be mutated, false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::mutate::Mutator;
    ///
    /// let mutator = Mutator::builder().build();
    ///
    /// let should_mutate = mutator.check_mutate();
    /// ```
    #[must_use]
    pub fn check_mutate(&self) -> bool {
        thread_rng().gen_range(0.0..1.0) < self.mutation_rate
    }

    /// Get the degree of mutation.
    ///
    /// # Returns
    ///
    /// The degree of mutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::mutate::Mutator;
    ///
    /// let mutator = Mutator::builder().build();
    ///
    /// let mutation_degree = mutator.mutation_size();
    /// ```
    #[must_use]
    pub fn mutation_size(&self) -> f64 {
        self.mutation_size * thread_rng().gen_range(-1.0..1.0)
    }

    /// Mutate a target.
    ///
    /// # Arguments
    ///
    /// - `target` - The target to mutate.
    ///
    /// # Returns
    ///
    /// The mutated target.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::mutate::Mutator;
    ///
    /// let mutator = Mutator::builder().build();
    ///
    /// let mut target = 0.0;
    /// target = mutator.mutate(target);
    /// ```
    pub fn mutate<TMutationTarget>(&self, tm: TMutationTarget) -> TMutationTarget
    where
        TMutationTarget: super::Target,
    {
        tm.mutate(self)
    }
}

/// A builder for a mutator.
///
/// # Examples
///
/// ```
/// use farm::mutate::Builder;
///
/// let mutator = Builder::default().build();
/// ```
pub struct Builder {
    mutation_rate: f64,
    mutation_size: f64,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            mutation_rate: 0.15,
            mutation_size: 0.15,
        }
    }
}

impl Builder {
    /// Set the mutation rate.
    ///
    /// # Arguments
    ///
    /// - `mutation_rate` - The new mutation rate, between 0.0 and 1.0.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::mutate::Builder;
    ///
    /// let mutator = Builder::default().mutation_rate(0.15).build();
    /// ```
    #[must_use]
    pub fn mutation_rate(mut self, mutation_rate: f64) -> Self {
        self.mutation_rate = mutation_rate;
        self
    }

    /// Set the mutation size.
    ///
    /// # Arguments
    ///
    /// - `mutation_size` - The new mutation size.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::mutate::Builder;
    ///
    /// let mutator = Builder::default().mutation_size(0.15).build();
    /// ```
    #[must_use]
    pub fn mutation_size(mut self, mutation_size: f64) -> Self {
        self.mutation_size = mutation_size;
        self
    }

    /// Build the mutator.
    ///
    /// # Returns
    ///
    /// The new mutator.
    ///
    /// # Examples
    ///
    /// ```
    /// use farm::mutate::Builder;
    ///
    /// let mutator = Builder::default().build();
    /// ```
    #[must_use]
    pub fn build(self) -> Mutator {
        Mutator {
            mutation_rate: self.mutation_rate,
            mutation_size: self.mutation_size,
        }
    }
}
