pub type Generation<TGenome> = Vec<TGenome>;

/// Stock a generation with genomes.
///
/// # Examples
///
/// ```
/// use evo::{Stock, Generation};
///
/// #[derive(Debug, PartialEq)]
/// struct Genome {
///     value: f64,
/// }
///
/// struct Stocker;
///
/// impl Stock<Genome> for Stocker {
///     fn base_genome(&self) -> Genome {
///         Genome { value: 1.0 }
///     }
///
///     fn generate(&self, base: &Genome) -> Genome {
///         Genome { value: base.value * 2.0 }
///     }
/// }
///
/// let stocker = Stocker;
/// let generation = stocker.stock(3);
/// let expected = vec![
///     Genome { value: 2.0 },
///     Genome { value: 2.0 },
///     Genome { value: 2.0 },
/// ];
///
/// assert_eq!(generation, expected);
/// ```
pub trait Stock<TGenome> {
    fn base_genome(&self) -> TGenome;

    fn generate(&self, base: &TGenome) -> TGenome;

    fn stock(&self, generation_size: usize) -> Generation<TGenome> {
        let base = self.base_genome();
        let mut output = vec![];
        let iter = std::iter::repeat_with(|| self.generate(&base));
        output.extend(iter.take(generation_size));
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stock() {
        let stocker = Stocker;
        let generation = stocker.stock(3);
        let expected = vec![Genome { value: 2.0 }; 3];

        assert_eq!(generation, expected);
    }

    #[derive(Debug, PartialEq, Copy, Clone)]
    struct Genome {
        value: f64,
    }

    struct Stocker;

    impl Stock<Genome> for Stocker {
        fn base_genome(&self) -> Genome {
            Genome { value: 1.0 }
        }

        fn generate(&self, base: &Genome) -> Genome {
            Genome {
                value: base.value * 2.0,
            }
        }
    }
}
