use crate::Neuron;

/// A layer of neurons.
pub struct Layer {
    neurons: Vec<Box<dyn Neuron>>,
}

impl Layer {
    /// Create a new layer of neurons.
    pub fn new(neurons: Vec<Box<dyn Neuron>>) -> Self {
        Self { neurons }
    }

    /// Activate the layer.
    pub fn activate(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|n| n.activate(inputs)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Sigmoid;
    use crate::neuron::BasicNeuronBuilder;

    #[test]
    fn test_layer() {
        let layer = Layer::new(vec![
            Box::new(
                BasicNeuronBuilder::default()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(Sigmoid)
                    .build(),
            ),
            Box::new(
                BasicNeuronBuilder::default()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(Sigmoid)
                    .build(),
            ),
        ]);

        let output = layer.activate(&[0.0, 0.0]);
        assert_eq!(output.len(), 2);
    }
}
