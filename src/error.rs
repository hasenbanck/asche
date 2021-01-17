use thiserror::Error;

/// Errors that asche can throw.
#[derive(Error, Debug)]
pub enum AscheError {
    /// A std::ffi::NulError.
    #[error(transparent)]
    NulError(#[from] std::ffi::NulError),
    /// A std::str::Utf8Error.
    #[error(transparent)]
    Utf8Error(#[from] std::str::Utf8Error),
    /// A ash::LoadingError.
    #[error(transparent)]
    LoadingError(#[from] ash::LoadingError),
    /// A ash::InstanceError.
    #[error(transparent)]
    InstanceError(#[from] ash::InstanceError),
    /// A ash::vk::Result.
    #[error(transparent)]
    VkResult(#[from] ash::vk::Result),

    /// The requested device type couldn't be acquired.
    #[error("can't acquire requested device")]
    DeviceAcquireError,

    /// An unspecified asche error.
    #[error("unknown asche error")]
    Unknown,
}
