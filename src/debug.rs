use std::ffi::CStr;

use ash::vk;
use tracing::{debug, error, info, warn};

/// Callback function for the debug utils logging.
pub(crate) unsafe extern "system" fn debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if std::thread::panicking() {
        return vk::FALSE;
    }

    let message = CStr::from_ptr((*p_callback_data).p_message);
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

/// Callback function for the debug report logging.
pub(crate) unsafe extern "system" fn debug_report_callback(
    report_flag: vk::DebugReportFlagsEXT,
    _: vk::DebugReportObjectTypeEXT,
    _object: u64,
    _location: usize,
    _msg_code: i32,
    layer_prefix: *const std::os::raw::c_char,
    description: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    if std::thread::panicking() {
        return vk::FALSE;
    }

    let layer_prefix = CStr::from_ptr(layer_prefix).to_str().unwrap();
    let description = CStr::from_ptr(description).to_str().unwrap();

    match report_flag {
        vk::DebugReportFlagsEXT::ERROR => {
            error!("[{}] {}", layer_prefix, description)
        }
        vk::DebugReportFlagsEXT::WARNING => {
            warn!("[{}] {}", layer_prefix, description)
        }
        vk::DebugReportFlagsEXT::INFORMATION => {
            info!("[{}] {}", layer_prefix, description)
        }
        vk::DebugReportFlagsEXT::DEBUG => {
            debug!("[{}] {}", layer_prefix, description)
        }
        _ => {
            warn!("[{}] {}", layer_prefix, description);
        }
    }

    vk::FALSE
}
