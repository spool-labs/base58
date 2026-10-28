//! What a caller gets back when input cannot be converted

use core::fmt;

/// Why a base58 string could not be turned into bytes
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// A byte that is not in the base58 alphabet
    InvalidCharacter(u8),

    /// More characters than the requested width can hold
    TooLong,

    /// Fewer characters than the requested width needs
    TooShort,

    /// Leading ones claim more zero bytes than the output has room for
    OutputTooLong,

    /// The value is too large for the requested width
    ValueTooLarge,
}

/// Why a batch of base58 strings could not be turned into bytes
///
/// A batch stops at the first input it cannot convert, and names it, because a
/// caller decoding a hundred keys from one request has to say which one was
/// wrong rather than that one of them was.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchError {
    /// The output holds fewer values than the input has
    OutputTooSmall,

    /// The input at this position could not be converted
    Input {
        /// Its position in the batch
        at: usize,

        /// Why it could not be converted
        error: DecodeError,
    },
}

/// Why bytes could not be turned into a base58 string
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodeError {
    /// The output buffer is smaller than the encoding needs
    OutputTooSmall,

    /// The input is longer than this codec converts in one piece
    InputTooLong,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, out: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DecodeError::InvalidCharacter(byte) => write!(out, "byte {byte} is not base58"),
            DecodeError::TooLong => out.write_str("too many characters for this width"),
            DecodeError::TooShort => out.write_str("too few characters for this width"),
            DecodeError::OutputTooLong => out.write_str("more leading zeros than the output holds"),
            DecodeError::ValueTooLarge => out.write_str("value does not fit this width"),
        }
    }
}

impl fmt::Display for BatchError {
    fn fmt(&self, out: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BatchError::OutputTooSmall => out.write_str("output holds fewer values than the input"),
            BatchError::Input { at, error } => write!(out, "input {at}: {error}"),
        }
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, out: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EncodeError::OutputTooSmall => out.write_str("output buffer is too small"),
            EncodeError::InputTooLong => out.write_str("input is too long to encode"),
        }
    }
}

impl core::error::Error for DecodeError {}

impl core::error::Error for BatchError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            BatchError::OutputTooSmall => None,
            BatchError::Input { error, .. } => Some(error),
        }
    }
}

impl core::error::Error for EncodeError {}
