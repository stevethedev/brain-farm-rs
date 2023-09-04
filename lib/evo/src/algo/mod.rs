mod run;
mod sort;

pub use self::sort::sort_generation;

/// Algorithms for evolving populations.
///
/// # Examples
///
/// ```
/// use evo::Algorithm;
///
/// let algorithm = Algorithm::builder().build();
/// ```
pub struct Algorithm {
    elitism: usize,
    tournament_size: usize,
}

impl Algorithm {
    /// Create a new algorithm builder.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::Algorithm;
    ///
    /// let algorithm = Algorithm::builder().build();
    /// ```
    #[must_use]
    pub fn builder() -> Builder {
        Builder::default()
    }
}

/// Builder for [`Algorithm`].
///
/// # Examples
///
/// ```
/// use evo::Algorithm;
///
/// let algorithm = Algorithm::builder()
///    .elitism(1)
///    .tournament_size(10)
///    .build();
/// ```
pub struct Builder {
    elitism: usize,
    tournament_size: usize,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            elitism: 1,
            tournament_size: 10,
        }
    }
}

impl Builder {
    /// Set the number of elite networks to keep.
    ///
    /// # Arguments
    ///
    /// - `elitism` - The number of elite networks to keep.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::Algorithm;
    ///
    /// let algorithm = Algorithm::builder()
    ///   .elitism(1)
    ///   .tournament_size(10)
    ///   .build();
    /// ```
    #[must_use]
    pub fn elitism(mut self, elitism: usize) -> Self {
        self.elitism = elitism;
        self
    }

    /// Set the tournament size.
    ///
    /// # Arguments
    ///
    /// - `tournament_size` - The tournament size.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::Algorithm;
    ///
    /// let algorithm = Algorithm::builder()
    ///     .elitism(1)
    ///     .tournament_size(10)
    ///     .build();
    /// ```
    #[must_use]
    pub fn tournament_size(mut self, tournament_size: usize) -> Self {
        self.tournament_size = tournament_size;
        self
    }

    /// Build the algorithm.
    ///
    /// # Returns
    ///
    /// The algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::Algorithm;
    ///
    /// let algorithm = Algorithm::builder()
    ///     .elitism(1)
    ///     .tournament_size(10)
    ///     .build();
    /// ```
    #[must_use]
    pub fn build(self) -> Algorithm {
        Algorithm {
            elitism: self.elitism,
            tournament_size: self.tournament_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm() {
        let algorithm = Algorithm::builder().build();
        assert_eq!(algorithm.elitism, 1);
        assert_eq!(algorithm.tournament_size, 10);
    }

    #[test]
    fn test_algorithm_builder() {
        let algorithm = Algorithm::builder().elitism(2).tournament_size(20).build();
        assert_eq!(algorithm.elitism, 2);
        assert_eq!(algorithm.tournament_size, 20);
    }
}
