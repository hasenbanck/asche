//! Implements the device context.

use ash::version::DeviceV1_0;
use ash::vk;

use crate::{Instance, Result};

/// The device context..
pub(crate) struct Context {
    pub(crate) instance: Instance,
    pub(crate) logical_device: ash::Device,
    pub(crate) physical_device: vk::PhysicalDevice,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { self.logical_device.destroy_device(None) };
    }
}

impl Context {
    /// Sets a debug name for an object.
    pub(crate) fn set_object_name(
        &self,
        name: &str,
        object_type: vk::ObjectType,
        object_handle: u64,
    ) -> Result<()> {
        let name = std::ffi::CString::new(name.to_owned())?;
        let info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_name(&name)
            .object_type(object_type)
            .object_handle(object_handle);
        unsafe {
            self.instance
                .debug_utils
                .debug_utils_set_object_name(self.logical_device.handle(), &info)?
        };
        Ok(())
    }
}
