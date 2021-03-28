use std::error::Error;

use erupt::vk;

/// Errors that asche can throw.
#[derive(Debug)]
pub enum AscheError {
    /// A std::io::Error.
    IoError(std::io::Error),
    /// A std::ffi::NulError.
    NulError(std::ffi::NulError),
    /// A std::str::Utf8Error.
    Utf8Error(std::str::Utf8Error),
    /// A erupt::utils::loading::EntryLoaderError.
    EntryLoaderError(erupt::utils::loading::EntryLoaderError),

    /// A erupt::LoaderError.
    LoaderError(erupt::LoaderError),

    /// A vk_alloc::AllocatorError.
    VkAllocError(vk_alloc::AllocatorError),

    /// A VKResult error.
    VkResult(vk::Result),

    /// Can't load the debug utils extension.
    DebugUtilsMissing,

    /// The requested device type couldn't be found.
    RequestDeviceError,

    /// Can't find a queue family.
    QueueFamilyNotFound(String),

    /// The selected format / color space for the swapchain is not supported by the device.
    SwapchainFormatIncompatible,

    /// The requested device feature couldn't be found.
    DeviceFeatureMissing,

    /// The selected presentation mode is unsupported.
    PresentationModeUnsupported,

    /// Swapchain is not initialized.
    SwapchainNotInitialized,

    /// The requested buffer has a size of zero.
    BufferZeroSize,

    /// No queue was configured.
    NoQueueConfigured,
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
            AscheError::EntryLoaderError(err) => {
                write!(f, "{:?}", err.source())
            }
            AscheError::LoaderError(err) => {
                write!(f, "{:?}", err.source())
            }
            AscheError::VkAllocError(err) => {
                write!(f, "{:?}", err.source())
            }
            AscheError::VkResult(err) => {
                write!(f, "vk::Result({})", err)
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
                write!(f, "the requested device feature couldn't be found")
            }
            AscheError::BufferZeroSize => {
                write!(f, "the requested buffer has a size of zero")
            }
            AscheError::NoQueueConfigured => {
                write!(f, "no queue was configured")
            }
        }
    }
}

impl std::error::Error for AscheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            AscheError::NulError(ref e) => Some(e),
            AscheError::Utf8Error(ref e) => Some(e),
            AscheError::EntryLoaderError(ref e) => Some(e),
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

impl From<erupt::LoaderError> for AscheError {
    fn from(err: erupt::LoaderError) -> AscheError {
        AscheError::LoaderError(err)
    }
}

impl From<vk::Result> for AscheError {
    fn from(err: vk::Result) -> AscheError {
        AscheError::VkResult(err)
    }
}

impl From<erupt::utils::loading::EntryLoaderError> for AscheError {
    fn from(err: erupt::utils::loading::EntryLoaderError) -> AscheError {
        AscheError::EntryLoaderError(err)
    }
}

impl From<vk_alloc::AllocatorError> for AscheError {
    fn from(err: vk_alloc::AllocatorError) -> AscheError {
        AscheError::VkAllocError(err)
    }
}
