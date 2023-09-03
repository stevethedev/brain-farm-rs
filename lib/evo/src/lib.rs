#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic
)]

mod algo;
mod fitness_calc;
mod genome;

pub use self::{
    algo::Algorithm,
    fitness_calc::{Compare, CompareRecord, FitnessCalc, Predict, TrainingRecord},
    genome::{Breeder as GenomeBreeder, BreederManager as GenomeBreederManager, Generation, Stock},
};
