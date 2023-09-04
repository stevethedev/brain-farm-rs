/// Breeder trait
///
/// # Examples
///
/// ```
/// use evo::Breed;
///
/// #[derive(Debug, PartialEq)]
/// struct Genome {
///    value: f64,
/// }
///
/// struct Breeder;
///
/// impl Breed<Genome> for Breeder {
///     fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
///         Genome {
///             value: (pair.0.value + pair.1.value) / 2.0,
///         }
///     }
/// }
///
/// let breeder = Breeder;
/// let left = Genome { value: 1.0 };
/// let right = Genome { value: 2.0 };
/// let offspring = breeder.crossover((&left, &right));
/// assert_eq!(offspring, Genome { value: 1.5 });
/// ```
pub trait Breed<TGenome> {
    /// Crossover two genomes.
    ///
    /// # Arguments
    ///
    /// - `pair` is the pair of genomes to crossover.
    ///
    /// # Returns
    ///
    /// The offspring genome.
    fn crossover(&self, pair: (&TGenome, &TGenome)) -> TGenome;

    /// Mutate a genome.
    ///
    /// # Arguments
    ///
    /// - `genome` is the genome to mutate.
    ///
    /// # Returns
    ///
    /// The mutated genome.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::Breed;
    ///
    /// #[derive(Debug, PartialEq)]
    /// struct Genome {
    ///   value: f64,
    /// }
    ///
    /// struct Breeder;
    ///
    /// impl Breed<Genome> for Breeder {
    ///     fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
    ///         Genome {
    ///             value: (pair.0.value + pair.1.value) / 2.0,
    ///         }
    ///     }
    ///
    ///     fn mutate(&self, mut genome: Genome) -> Genome {
    ///         genome.value *= 2.0;
    ///         genome
    ///     }
    /// }
    ///
    /// let breeder = Breeder;
    /// let genome = Genome { value: 1.0 };
    /// let mutated = breeder.mutate(genome);
    /// assert_eq!(mutated, Genome { value: 2.0 });
    /// ```
    fn mutate(&self, genome: TGenome) -> TGenome {
        genome
    }

    /// Convert this breeder into a manager.
    ///
    /// # Returns
    ///
    /// The manager.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::Breed;
    ///
    /// #[derive(Debug, PartialEq)]
    /// struct Genome {
    ///  value: f64,
    /// }
    ///
    /// struct Breeder;
    ///
    /// impl Breed<Genome> for Breeder {
    ///    fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
    ///         Genome {
    ///             value: (pair.0.value + pair.1.value) / 2.0,
    ///         }
    ///     }
    /// }
    ///
    /// let manager = Breeder.to_manager();
    /// let left = Genome { value: 1.0 };
    /// let right = Genome { value: 2.0 };
    /// let offspring = manager.breed(&left, &right);
    /// assert_eq!(offspring, Genome { value: 1.5 });
    /// ```
    fn to_manager(self) -> Manager<TGenome, Self>
    where
        Self: Sized,
    {
        Manager::new(self)
    }
}

/// A manager for breeding genomes.
///
/// # Examples
///
/// ```
/// use evo::Breed;
///
/// #[derive(Debug, PartialEq)]
/// struct Genome {
///   value: f64,
/// }
///
/// struct Breeder;
///
/// impl Breed<Genome> for Breeder {
///    fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
///       Genome {
///         value: (pair.0.value + pair.1.value) / 2.0,
///      }
///   }
/// }
///
/// let manager = Breeder.to_manager();
/// let left = Genome { value: 1.0 };
/// let right = Genome { value: 2.0 };
/// let offspring = manager.breed(&left, &right);
/// assert_eq!(offspring, Genome { value: 1.5 });
/// ```
pub struct Manager<TGenome, TBreeder>
where
    TBreeder: Breed<TGenome>,
{
    _phantom: std::marker::PhantomData<TGenome>,
    breeder: TBreeder,
}

impl<TGenome, TBreeder> Manager<TGenome, TBreeder>
where
    TBreeder: Breed<TGenome>,
{
    /// Create a new manager.
    ///
    /// # Arguments
    ///
    /// - `breeder` is the breeder to use.
    ///
    /// # Returns
    ///
    /// The manager.
    pub fn new(breeder: TBreeder) -> Self {
        Self {
            breeder,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Breed two genomes.
    ///
    /// # Arguments
    ///
    /// - `left` is the left genome.
    /// - `right` is the right genome.
    ///
    /// # Returns
    ///
    /// The offspring genome.
    ///
    /// # Examples
    ///
    /// ```
    /// use evo::Breed;
    ///
    /// #[derive(Debug, PartialEq)]
    /// struct Genome {
    ///  value: f64,
    /// }
    ///
    /// struct Breeder;
    ///
    /// impl Breed<Genome> for Breeder {
    ///     fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
    ///         Genome {
    ///             value: (pair.0.value + pair.1.value) / 2.0,
    ///         }
    ///     }
    /// }
    ///
    /// let manager = Breeder.to_manager();
    /// let left = Genome { value: 1.0 };
    /// let right = Genome { value: 2.0 };
    /// let offspring = manager.breed(&left, &right);
    /// assert_eq!(offspring, Genome { value: 1.5 });
    /// ```
    pub fn breed(&self, left: &TGenome, right: &TGenome) -> TGenome {
        let offspring = self.breeder.crossover((left, right));
        self.breeder.mutate(offspring)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_breed() {
        #[derive(Debug, PartialEq)]
        struct Genome {
            value: f64,
        }

        struct Breeder;

        impl Breed<Genome> for Breeder {
            fn crossover(&self, pair: (&Genome, &Genome)) -> Genome {
                Genome {
                    value: (pair.0.value + pair.1.value) / 2.0,
                }
            }
        }

        let manager = Manager::new(Breeder);

        let left = Genome { value: 1.0 };
        let right = Genome { value: 2.0 };
        let offspring = manager.breed(&left, &right);
        assert_eq!(offspring, Genome { value: 1.5 });
    }
}
