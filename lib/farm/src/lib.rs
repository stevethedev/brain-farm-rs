pub mod breed;
pub mod genome;
pub mod mutate;

pub fn run() {
    use genome::Generate;

    let neuron_config = genome::neuron::GenerateConfig {
        activator_generator: || genome::activator::Genome::generate(()),
        weight_generator: || {
            std::iter::from_fn(|| Some(f64::generate(-1.0..=1.0)))
                .take(3)
                .collect()
        },
        bias_generator: || f64::generate(-2.0..=2.0),
    };

    let layer_config = genome::layer::GenerateConfig {
        neuron_generator: || {
            std::iter::from_fn(|| Some(genome::neuron::Genome::generate(&neuron_config)))
                .take(4)
                .collect()
        },
    };

    let network_config = genome::network::GenerateConfig {
        layer_generator: || {
            std::iter::from_fn(|| Some(genome::layer::Genome::generate(&layer_config)))
                .take(5)
                .collect()
        },
    };

    let genome = genome::network::Genome::generate(&network_config);

    println!("{:?}", genome);
}
