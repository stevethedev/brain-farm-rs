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

pub use self::{
    algo::Algorithm,
    fitness_calc::{Compare, CompareRecord, FitnessCalc, Predict, TrainingRecord},
};
