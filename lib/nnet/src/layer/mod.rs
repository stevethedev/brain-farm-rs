use crate::Neuron;

/// A layer of neurons.
pub struct Layer {
    neurons: Vec<Box<dyn Neuron>>,
}

impl Layer {
    /// Create a new builder.
    ///
    /// # Returns
    ///
    /// A new builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::Layer;
    ///
    /// let layer = Layer::builder().build();
    /// ```
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Activate the layer.
    ///
    /// # Arguments
    ///
    /// - `inputs` to activate the layer with.
    ///
    /// # Returns
    ///
    /// The output of the layer.
    pub fn activate(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|n| n.activate(inputs)).collect()
    }

    /// Get a reference to the set of neurons.
    ///
    /// # Returns
    ///
    /// A reference to the set of neurons.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    ///
    /// assert_eq!(layer.neurons().len(), 1);
    /// ```
    pub fn neurons(&self) -> &[Box<dyn Neuron>] {
        &self.neurons
    }
}

/// A builder for `Layer`s.
///
/// # Examples
///
/// ```
/// use nnet::{Layer, BasicNeuron};
///
/// let neuron = BasicNeuron::builder().build();
/// let layer = Layer::builder().add_neuron(neuron).build();
/// ```
#[derive(Default)]
pub struct Builder {
    neurons: Vec<Box<dyn Neuron>>,
}

impl Builder {
    /// Add a neuron to the layer.
    ///
    /// # Arguments
    ///
    /// - `neuron` to add to the layer.
    ///
    /// # Returns
    ///
    /// The builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::{Layer, BasicNeuron};
    ///
    /// let neuron = BasicNeuron::builder().build();
    /// let layer = Layer::builder().add_neuron(neuron).build();
    ///
    /// assert_eq!(layer.neurons().len(), 1);
    /// ```
    pub fn add_neuron(mut self, neuron: impl Into<Box<dyn Neuron>>) -> Self {
        self.neurons.push(neuron.into());
        self
    }

    /// Build the layer.
    ///
    /// # Returns
    ///
    /// The built layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::Layer;
    ///
    /// let layer = Layer::builder().build();
    /// ```
    pub fn build(self) -> Layer {
        let Self { neurons } = self;
        Layer { neurons }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{activation::Sigmoid, BasicNeuron};

    // Define a basic neuron implementation for testing.
    struct TestNeuron;

    impl Neuron for TestNeuron {
        fn activate(&self, inputs: &[f64]) -> f64 {
            inputs.iter().sum()
        }
    }

    #[test]
    fn test_layer() {
        let layer = Layer::builder()
            .add_neuron(
                BasicNeuron::builder()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(Sigmoid)
                    .build(),
            )
            .add_neuron(
                BasicNeuron::builder()
                    .bias(0.0)
                    .weights(vec![0.0, 0.0])
                    .activation(Sigmoid)
                    .build(),
            )
            .build();

        let output = layer.activate(&[0.0, 0.0]);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_layer_builder() {
        // Create a layer using the builder and check if the number of neurons is correct.
        let layer = Layer::builder()
            .add_neuron(TestNeuron)
            .add_neuron(TestNeuron)
            .build();

        assert_eq!(layer.neurons().len(), 2);
    }

    #[test]
    fn test_layer_activation() {
        // Create a layer with two neurons and activate it with some inputs.
        let layer = Layer::builder()
            .add_neuron(TestNeuron)
            .add_neuron(TestNeuron)
            .build();

        let inputs = vec![1.0, 2.0];
        let output = layer.activate(&inputs);

        // The output should have the same number of values as neurons.
        assert_eq!(output.len(), 2);

        // Check if the activation results are as expected (inputs summed up).
        assert_eq!(output[0], 3.0);
        assert_eq!(output[1], 3.0);
    }
}
