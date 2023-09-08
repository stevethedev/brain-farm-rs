use crate::genome::Generate;
pub use evo::Stock;

/// A stocker for the given genome using the given config.
///
/// # Examples
///
/// ```
/// use farm::genome::activator::{Genome, Gene};
/// use farm::stock::{Stock, Stocker};
///
/// let stocker = Stocker::<_, Genome>::new(|| Gene::Linear);
/// let genome = stocker.generate();
/// let generation = stocker.stock(3);
/// let expected = vec![
///     Genome { activator: Gene::Linear },
///     Genome { activator: Gene::Linear },
///     Genome { activator: Gene::Linear },
/// ];
///
/// assert_eq!(genome, Genome { activator: Gene::Linear });
/// assert_eq!(generation, expected);
/// ```
pub struct Stocker<TConfig, TGenome>
where
    TGenome: for<'a> Generate<&'a TConfig>,
{
    _phantom: std::marker::PhantomData<TGenome>,
    base_config: TConfig,
}

impl<TConfig, TGenome> Stocker<TConfig, TGenome>
where
    TGenome: for<'a> Generate<&'a TConfig>,
{
    pub fn new(base_config: TConfig) -> Self {
        Self {
            base_config,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<TConfig, TGenome> Stock<TGenome> for Stocker<TConfig, TGenome>
where
    TGenome: for<'a> Generate<&'a TConfig>,
{
    fn generate(&self) -> TGenome {
        TGenome::generate(&self.base_config)
    }
}
