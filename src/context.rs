//! Implements the device context.
use ash::version::DeviceV1_0;
use ash::vk;
use parking_lot::Mutex;

use crate::{Instance, Result};

/// The Vulkan context.
pub struct Context {
    /// The wrapped Vulkan instance.
    pub instance: Instance,
    /// The raw logical Vulkan device.
    pub logical_device: ash::Device,
    /// The raw physical Vulkan device.
    pub physical_device: vk::PhysicalDevice,
    /// The memory allocator.
    pub allocator: Mutex<vk_alloc::Allocator>,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.allocator.lock().free_all();
        unsafe { self.logical_device.destroy_device(None) };
    }
}

impl Context {
    /// Sets a debug name for an object.
    #[cfg(debug_assertions)]
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

    /// Sets a debug name for an object.
    #[cfg(not(debug_assertions))]
    pub(crate) fn set_object_name(
        &self,
        _name: &str,
        _object_type: vk::ObjectType,
        _object_handle: u64,
    ) -> Result<()> {
        Ok(())
    }
}
