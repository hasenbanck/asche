use std::ffi::c_void;
use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing::error;

use crate::context::Context;
use crate::{AscheError, Result};

/// Wraps an acceleration structure.
#[derive(Debug)]
pub struct AccelerationStructure {
    /// The raw Vulkan acceleration structure.
    pub raw: vk::AccelerationStructureKHR,
    context: Arc<Context>,
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_acceleration_structure_khr(Some(self.raw), None);
        };
    }
}

impl AccelerationStructure {
    pub(crate) fn new(raw: vk::AccelerationStructureKHR, context: Arc<Context>) -> Self {
        Self { raw, context }
    }

    /// Query an address of a acceleration structure.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetAccelerationStructureDeviceAddressKHR.html)"]
    pub fn device_address(&self) -> vk::DeviceAddress {
        let info = vk::AccelerationStructureDeviceAddressInfoKHRBuilder::new()
            .acceleration_structure(self.raw);
        unsafe {
            self.context
                .device
                .get_acceleration_structure_device_address_khr(&info)
        }
    }

    /// Query acceleration structure meta-data on the host.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkWriteAccelerationStructuresPropertiesKHR.html)"]
    pub fn write_acceleration_structures_properties(
        &self,
        query_type: vk::QueryType,
        data: &[u8],
        stride: usize,
    ) -> Result<()> {
        let acceleration_structures = [self.raw];
        #[allow(clippy::as_conversions)]
        unsafe {
            self.context
                .device
                .write_acceleration_structures_properties_khr(
                    &acceleration_structures,
                    query_type,
                    data.len(),
                    data.as_ptr() as *mut c_void,
                    stride,
                )
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!(
                "Unable to query acceleration structure meta-data on the host: {}",
                err
            );
            AscheError::VkResult(err)
        })
    }
}
