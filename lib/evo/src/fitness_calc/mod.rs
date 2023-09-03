mod calc;
mod compare;
mod error;
mod predict;
mod training;

pub use self::{
    calc::{Builder, Calc as FitnessCalc},
    compare::{Compare, Record as CompareRecord},
    error::{Error, Result},
    predict::Predict,
    training::Record as TrainingRecord,
};
