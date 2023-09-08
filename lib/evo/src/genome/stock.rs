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
/// struct Stocker(Genome);
///
/// impl Stock<Genome> for Stocker {
///     fn generate(&self) -> Genome {
///         Genome { value: self.0.value * 2.0 }
///     }
/// }
///
/// let stocker = Stocker(Genome { value: 1.0 });
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
    fn generate(&self) -> TGenome;

    fn stock(&self, generation_size: usize) -> super::Generation<TGenome> {
        std::iter::repeat_with(|| self.generate())
            .take(generation_size)
            .collect()
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
        fn generate(&self) -> Genome {
            Genome { value: 2.0 }
        }
    }
}
