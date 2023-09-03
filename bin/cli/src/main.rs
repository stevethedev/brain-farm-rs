#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic
)]

use nnet::Network;

const NETWORK_JSON: &str = r#"
{
    "layers": [
        {
            "neurons": [
                {
                    "Basic": {
                        "bias": 0.0,
                        "weights": [
                            1.0
                        ],
                        "activation": {
                            "Sigmoid": null
                        }
                    }
                }
            ]
        },
        {
            "neurons": [
                {
                    "Basic": {
                        "bias": 0.0,
                        "weights": [
                            0.5
                        ],
                        "activation": {
                            "Sigmoid": null
                        }
                    }
                }
            ]
        }
    ]
}
"#;

fn main() {
    let network = Network::parse_json(NETWORK_JSON).unwrap();

    let input = vec![1.0, 1.0];
    let output = network.activate(&input);

    println!("{network:?} took {input:?} and produced {output:?}");
}
