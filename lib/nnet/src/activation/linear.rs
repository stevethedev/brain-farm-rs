use super::ActivationFunction;

pub struct Linear;

impl ActivationFunction for Linear {
    fn activate(&self, x: f64) -> f64 {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let expected: f64 = 0.1 * 0.1 + 0.5 * 0.2 + 1.0 * 0.3 + 1.5 * 0.4 + 1.0;

        let lin = Linear;
        let outputs = lin.activate(0.1 * 0.1 + 0.5 * 0.2 + 1.0 * 0.3 + 1.5 * 0.4 + 1.0);
        assert!(
            (outputs - expected).abs() < f64::EPSILON,
            "Expected {} to be close to {}",
            outputs,
            expected
        );
    }
}
