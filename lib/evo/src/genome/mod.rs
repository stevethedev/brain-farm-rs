mod breeder;
mod stock;

pub type Generation<TGenome> = Vec<TGenome>;

pub use self::{
    breeder::{Breeder, Manager as BreederManager},
    stock::Stock,
};
