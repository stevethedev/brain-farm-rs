mod linear;
mod sigmoid;

pub use linear::Linear;
pub use sigmoid::Sigmoid;

/// [`Neuron`] activation function.
///
/// # Examples
///
/// ```
/// use nnet::ActivationFunction;
///
/// let lin = ActivationFunction::linear();
/// ```
#[derive(Debug, PartialEq)]
pub enum Function {
    Linear(Linear),
    Sigmoid(Sigmoid),
}

impl Activate for Function {
    /// Activate the function.
    ///
    /// # Arguments
    ///
    /// - `input` to activate the function with.
    ///
    /// # Returns
    ///
    /// The output of the function.
    ///
    /// # Examples
    ///
    /// ```
    /// use nnet::ActivationFunction;
    ///
    /// let lin = ActivationFunction::linear();
    fn activate(&self, input: f64) -> f64 {
        match self {
            Self::Linear(lin) => lin.activate(input),
            Self::Sigmoid(sig) => sig.activate(input),
        }
    }
}

/// Trait for executing an activation function.
pub trait Activate {
    /// Activate the function.
    ///
    /// # Arguments
    ///
    /// - `input` to activate the function with.
    ///
    /// # Returns
    ///
    /// The output of the function.
    fn activate(&self, input: f64) -> f64;
}
