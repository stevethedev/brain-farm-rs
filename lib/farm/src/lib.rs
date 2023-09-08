pub mod breed;
pub mod genome;
pub mod mutate;
pub mod stock;

use crate::genome::Create;
use crate::stock::Stock;
use evo::Breed;

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
    let network_stocker = stock::Stocker::<_, genome::network::Genome>::new(network_config);

    let mutator = mutate::Mutator::builder()
        .mutation_size(0.015)
        .mutation_rate(0.15)
        .build();
    let breeder = breed::Breeder::new(mutator);

    let genome = network_stocker.generate();
    let genome = breeder.mutate(genome);
    let network = genome.create();

    println!("{:?}", genome);
    println!("{:?}", network);

    let output = network.activate(&[1.0, 2.0, 3.0, 4.0]);
    println!("{:?}", output);
}
