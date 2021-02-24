use std::error::Error;

use ash::vk;

/// Errors that asche can throw.
#[derive(Debug)]
pub enum AscheError {
    /// A std::io::Error.
    IoError(std::io::Error),
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
    /// A vk_alloc::AllocatorError.
    VkAllocError(vk_alloc::AllocatorError),

    /// Can't load the debug utils extension.
    DebugUtilsMissing,

    /// The requested device type couldn't be found.
    RequestDeviceError,

    /// Can't find a queue family.
    QueueFamilyNotFound(String),

    /// The selected format / color space for the swapchain is not supported by the device.
    SwapchainFormatIncompatible,

    /// The requested feature couldn't be found.
    DeviceFeatureMissing,

    /// The selected presentation mode is unsupported.
    PresentationModeUnsupported,

    /// Swapchain is not initialized.
    SwapchainNotInitialized,

    /// An unspecified asche error.
    Unspecified(String),
}

impl std::fmt::Display for AscheError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AscheError::IoError(err) => {
                write!(f, "{:?}", err.source())
            }
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
            AscheError::VkAllocError(err) => {
                write!(f, "{:?}", err.source())
            }
            AscheError::DebugUtilsMissing => {
                write!(f, "can't load the debug utils extension")
            }
            AscheError::RequestDeviceError => {
                write!(f, "can't find device with requested capabilities")
            }
            AscheError::QueueFamilyNotFound(family) => {
                write!(f, "can't find queue family: {}", family)
            }
            AscheError::SwapchainFormatIncompatible => {
                write!(f, "selected format / color space for the swapchain is not supported by the device")
            }
            AscheError::SwapchainNotInitialized => {
                write!(f, "swapchain is not initialized")
            }
            AscheError::PresentationModeUnsupported => {
                write!(f, "the selected presentation mode is unsupported")
            }
            AscheError::DeviceFeatureMissing => {
                write!(f, "the requested feature couldn't be found")
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

impl From<std::io::Error> for AscheError {
    fn from(err: std::io::Error) -> AscheError {
        AscheError::IoError(err)
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

impl From<(Vec<vk::Pipeline>, ash::vk::Result)> for AscheError {
    fn from(err: (Vec<vk::Pipeline>, vk::Result)) -> Self {
        AscheError::VkResult(err.1)
    }
}

impl From<vk_alloc::AllocatorError> for AscheError {
    fn from(err: vk_alloc::AllocatorError) -> AscheError {
        AscheError::VkAllocError(err)
    }
}
