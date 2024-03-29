//! Implements the internal context.
use erupt::vk;
#[cfg(feature = "tracing")]
use tracing1::error;

use crate::{AscheError, Instance, Result};

/// The internal context.
#[derive(Debug)]
pub(crate) struct Context {
    /// The raw logical Vulkan device.
    pub(crate) device: erupt::DeviceLoader,
    /// The raw physical Vulkan device.
    pub(crate) physical_device: vk::PhysicalDevice,
    /// The wrapped Vulkan instance.
    pub(crate) instance: Instance,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        };
    }
}

impl Context {
    /// Creates a new context.
    pub(crate) fn new(
        instance: Instance,
        device: erupt::DeviceLoader,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        Self {
            device,
            physical_device,
            instance,
        }
    }

    /// Sets a debug name for an object.
    #[cfg(debug_assertions)]
    pub(crate) fn set_object_name(
        &self,
        name: &str,
        object_type: vk::ObjectType,
        object_handle: u64,
    ) -> Result<()> {
        let name = std::ffi::CString::new(name.to_owned())?;
        let info = vk::DebugUtilsObjectNameInfoEXTBuilder::new()
            .object_name(&name)
            .object_type(object_type)
            .object_handle(object_handle);
        unsafe {
            self.device
                .set_debug_utils_object_name_ext(&info)
                .map_err(|err| {
                    #[cfg(feature = "tracing")]
                    error!("Unable to set the debug object name: {}", err);
                    AscheError::VkResult(err)
                })
        }?;

        Ok(())
    }

    /// Sets a debug name for an object.
    #[cfg(not(debug_assertions))]
    pub(crate) unsafe fn set_object_name(
        &self,
        _name: &str,
        _object_type: vk::ObjectType,
        _object_handle: u64,
    ) -> Result<()> {
        Ok(())
    }
}
