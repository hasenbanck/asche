//! Implements the debug callback.

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing1::{debug, error, info, warn};

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

    #[cfg(feature = "tracing")]
    {
        match message_severity {
            vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT => {
                error!("{:?} - {:?}", message_types, message)
            }
            vk::DebugUtilsMessageSeverityFlagBitsEXT::WARNING_EXT => {
                warn!("{:?} - {:?}", message_types, message)
            }
            vk::DebugUtilsMessageSeverityFlagBitsEXT::INFO_EXT => {
                info!("{:?} - {:?}", message_types, message)
            }
            vk::DebugUtilsMessageSeverityFlagBitsEXT::VERBOSE_EXT => {
                debug!("{:?} - {:?}", message_types, message)
            }
            _ => {
                warn!("{:?} - {:?}", message_types, message);
            }
        }
    }

    #[cfg(not(feature = "tracing"))]
    {
        match message_severity {
            vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT => {
                println!("ERROR: {:?} - {:?}", message_types, message)
            }
            vk::DebugUtilsMessageSeverityFlagBitsEXT::WARNING_EXT => {
                println!("WARN: {:?} - {:?}", message_types, message)
            }
            vk::DebugUtilsMessageSeverityFlagBitsEXT::INFO_EXT => {
                println!("INFO: {:?} - {:?}", message_types, message)
            }
            vk::DebugUtilsMessageSeverityFlagBitsEXT::VERBOSE_EXT => {
                println!("DEBUG: {:?} - {:?}", message_types, message)
            }
            _ => {
                println!("WARN: {:?} - {:?}", message_types, message);
            }
        }
    }

    vk::FALSE
}
