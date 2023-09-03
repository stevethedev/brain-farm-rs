/// An error that can occur when calculating fitness.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("cannot convert")]
    CannotConvert,

    #[error("result is NaN")]
    ResultNaN,

    #[error("result is infinite")]
    ResultInfinite,
}

/// A result that can occur when calculating fitness.
pub type Result<T> = std::result::Result<T, Error>;
