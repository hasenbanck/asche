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

    /// Sets a debug tag for an object.
    pub(crate) fn set_object_tag(
        &self,
        tag_name: TagName,
        tag_value: &[u8],
        object_type: vk::ObjectType,
        object_handle: u64,
    ) -> Result<()> {
        let info = vk::DebugUtilsObjectTagInfoEXT::builder()
            .tag_name(tag_name as u64)
            .tag(tag_value)
            .object_type(object_type)
            .object_handle(object_handle);
        unsafe {
            self.instance
                .debug_utils
                .debug_utils_set_object_tag(self.logical_device.handle(), &info)?
        };

        Ok(())
    }
}

/// Tag names used for debugging.
#[derive(Copy, Clone)]
pub(crate) enum TagName {
    /// Buffer pool index (u64)
    BufferPoolIndex = 0,
    /// Type of a queue (u8)
    QueueType,
}
