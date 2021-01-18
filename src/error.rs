use std::error::Error;

/// Errors that asche can throw.
#[derive(Debug)]
pub enum AscheError {
    /// A std::ffi::NulError.
    NulError(std::ffi::NulError),
    /// A std::str::Utf8Error.
    Utf8Error(std::str::Utf8Error),
    /// A ash::LoadingError.
    LoadingError(ash::LoadingError),
    /// A ash::InstanceError.
    InstanceError(ash::InstanceError),
    /// A ash::vk::Result.
    VkResult(ash::vk::Result),

    /// The requested device type couldn't be found.
    RequestDeviceError,

    /// Can't find a queue family.
    QueueFamilyNotFound(&'static str),

    /// An unspecified asche error.
    Unspecified(String),
}

impl std::fmt::Display for AscheError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AscheError::NulError(err) => {
                write!(f, "{:?}", err.source())
            }
            AscheError::Utf8Error(err) => {
                write!(f, "{:?}", err.source())
            }
            AscheError::LoadingError(err) => {
                write!(f, "{:?}", err.source())
            }
            AscheError::InstanceError(err) => {
                write!(f, "{:?}", err.source())
            }
            AscheError::VkResult(err) => {
                write!(f, "{:?}", err.source())
            }
            AscheError::RequestDeviceError => {
                write!(f, "can't find device with requested capabilities")
            }
            AscheError::QueueFamilyNotFound(family) => {
                write!(f, "can't find queue family: {}", family)
            }
            AscheError::Unspecified(message) => {
                write!(f, "{}", message)
            }
        }
    }
}

impl std::error::Error for AscheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            AscheError::NulError(ref e) => Some(e),
            AscheError::Utf8Error(ref e) => Some(e),
            AscheError::LoadingError(ref e) => Some(e),
            AscheError::InstanceError(ref e) => Some(e),
            AscheError::VkResult(ref e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::ffi::NulError> for AscheError {
    fn from(err: std::ffi::NulError) -> AscheError {
        AscheError::NulError(err)
    }
}

impl From<std::str::Utf8Error> for AscheError {
    fn from(err: std::str::Utf8Error) -> AscheError {
        AscheError::Utf8Error(err)
    }
}

impl From<ash::LoadingError> for AscheError {
    fn from(err: ash::LoadingError) -> AscheError {
        AscheError::LoadingError(err)
    }
}

impl From<ash::InstanceError> for AscheError {
    fn from(err: ash::InstanceError) -> AscheError {
        AscheError::InstanceError(err)
    }
}

impl From<ash::vk::Result> for AscheError {
    fn from(err: ash::vk::Result) -> AscheError {
        AscheError::VkResult(err)
    }
}
