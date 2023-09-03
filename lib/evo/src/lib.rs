#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic
)]

mod algo;
mod breeder;
mod fitness_calc;
mod stocker;

pub use self::{
    algo::Algorithm,
    breeder::{Breeder as GenomeBreeder, Manager as BreederManager},
    fitness_calc::{Compare, CompareRecord, FitnessCalc, Predict, TrainingRecord},
    stocker::{Generation, Stock},
};
