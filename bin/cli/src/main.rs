#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic
)]

use nnet::{Layer, Network, Neuron};

fn main() {
    let layers = vec![
        Layer::builder()
            .add_neuron(Neuron::basic().add_weight(1.0).build())
            .build(),
        Layer::builder()
            .add_neuron(Neuron::basic().add_weight(0.5).build())
            .build(),
    ];
    let network = Network::builder().add_layers(layers).build();

    let input = vec![1.0, 1.0];
    let output = network.activate(&input);

    println!("{network:?} took {input:?} and produced {output:?}");
}
