#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic
)]

mod algo;
mod breed;
mod fitness_calc;
mod genome;

pub use self::{
    algo::Algorithm as EvoAlgorithm,
    breed::{Breed, Manager as BreedManager},
    fitness_calc::{Compare, CompareRecord, FitnessCalc, Predict, TrainingRecord},
    genome::{Generation, Stock},
};
