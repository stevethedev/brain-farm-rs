/// The gene for an activation function.
///
/// # Examples
///
/// ```
/// use farm::genome::activator::Gene;
///
/// let gene = Gene::Linear;
/// ```
pub struct Genome {
    activator: Gene,
}

impl Genome {
    /// Create a new activation function gene.
    pub fn new(activator: Gene) -> Self {
        Self { activator }
    }

    /// Get the activation function.
    pub fn activator(&self) -> &Gene {
        &self.activator
    }

    /// Set the activation function.
    pub fn set_activator(&mut self, activator: Gene) {
        self.activator = activator;
    }
}

/// The gene for an activation function.
#[derive(Clone, Debug, PartialEq)]
pub enum Gene {
    /// Linear activation function.
    Linear,

    /// Sigmoid activation function.
    Sigmoid,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let gene = Gene::Linear;
        let genome = Genome::new(gene.clone());
        assert_eq!(genome.activator(), &gene);
    }

    #[test]
    fn test_activator() {
        let gene = Gene::Linear;
        let mut genome = Genome::new(gene.clone());
        assert_eq!(genome.activator(), &gene);

        let gene = Gene::Sigmoid;
        genome.set_activator(gene.clone());
        assert_eq!(genome.activator(), &gene);
    }
}
