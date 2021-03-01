//! Implements the debug callback.

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, error, info, warn};

/// Callback function for the debug utils logging.
pub(crate) unsafe extern "system" fn debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if std::thread::panicking() {
        return vk::FALSE;
    }

    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let ty = format!("{:?}", message_types);

    #[cfg(feature = "tracing")]
    {
        match message_severity {
            vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT => {
                error!("{} - {:?}", ty, message)
            }
            vk::DebugUtilsMessageSeverityFlagBitsEXT::WARNING_EXT => {
                warn!("{} - {:?}", ty, message)
            }
            vk::DebugUtilsMessageSeverityFlagBitsEXT::INFO_EXT => {
                info!("{} - {:?}", ty, message)
            }
            vk::DebugUtilsMessageSeverityFlagBitsEXT::VERBOSE_EXT => {
                debug!("{} - {:?}", ty, message)
            }
            _ => {
                warn!("{} - {:?}", ty, message);
            }
        }
    }

    #[cfg(not(feature = "tracing"))]
    {
        match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
                println!("ERROR: {} - {:?}", ty, message)
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
                println!("WARN: {} - {:?}", ty, message)
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
                println!("INFO: {} - {:?}", ty, message)
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
                println!("DEBUG: {} - {:?}", ty, message)
            }
            _ => {
                println!("WARN: {} - {:?}", ty, message);
            }
        }
    }

    vk::FALSE
}
