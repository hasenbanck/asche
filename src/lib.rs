#![warn(missing_docs)]
//! Provides an abstraction layer above ash to easier use Vulkan in Rust with minimal dependencies.

#[cfg(feature = "tracing")]
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, error, info, warn};

pub use {
    device::{Device, DeviceDescriptor, QueuePriorityDescriptor},
    error::AscheError,
    instance::{Instance, InstanceDescriptor},
};

pub(crate) mod device;
pub(crate) mod error;
pub(crate) mod instance;
pub(crate) mod swapchain;

pub(crate) type Result<T> = std::result::Result<T, AscheError>;

/// Construct a `*const std::os::raw::c_char` from a string
#[macro_export]
macro_rules! cstr {
    ($s:expr) => {
        concat!($s, "\0") as *const str as *const c_char
    };
}

/// Callback function for the debug utils logging.
#[cfg(feature = "tracing")]
pub(crate) unsafe extern "system" fn debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if std::thread::panicking() {
        return vk::FALSE;
    }

    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let ty = format!("{:?}", message_type);

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!("{} - {:?}", ty, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!("{} - {:?}", ty, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!("{} - {:?}", ty, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            debug!("{} - {:?}", ty, message)
        }
        _ => {
            warn!("{} - {:?}", ty, message);
        }
    }

    vk::FALSE
}
