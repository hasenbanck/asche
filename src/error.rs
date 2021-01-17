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

    /// The requested device type couldn't be found.
    #[error("can't find device with requested capabilities")]
    RequestDeviceError,

    /// Can't find a queue family.
    #[error("can't find queue family: {0}")]
    QueueFamilyNotFound(&'static str),

    /// An unspecified asche error.
    #[error("{0}")]
    Unspecified(String),
}
