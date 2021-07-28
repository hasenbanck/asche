use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing1::error;

use crate::context::Context;
use crate::{AscheError, Result};

/// Wraps a deferred operation.
#[derive(Debug)]
pub struct DeferredOperation {
    raw: vk::DeferredOperationKHR,
    context: Arc<Context>,
}

impl Drop for DeferredOperation {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_deferred_operation_khr(Some(self.raw), None);
        };
    }
}

impl DeferredOperation {
    pub(crate) fn new(raw: vk::DeferredOperationKHR, context: Arc<Context>) -> Self {
        Self { raw, context }
    }

    /// The raw Vulkan deferred operation handle.
    #[inline]
    pub fn raw(&self) -> vk::DeferredOperationKHR {
        self.raw
    }

    /// Assign a thread to a deferred operation.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDeferredOperationJoinKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn join(&self) -> Result<()> {
        self.context
            .device
            .deferred_operation_join_khr(self.raw)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to assign a thread to a deferred operation: {}", err);
                AscheError::VkResult(err)
            })
    }

    /// Query the maximum concurrency on a deferred operation.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetDeferredOperationMaxConcurrencyKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn max_concurrency(&self) -> u32 {
        self.context
            .device
            .get_deferred_operation_max_concurrency_khr(self.raw)
    }

    /// Query the result of a deferred operation.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetDeferredOperationResultKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn result(&self) -> Result<()> {
        self.context
            .device
            .get_deferred_operation_result_khr(self.raw)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!(
                    "Unable to query the result of a deferred operation: {}",
                    err
                );
                AscheError::VkResult(err)
            })
    }

    /// Build an acceleration structure on the host.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkBuildAccelerationStructuresKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn build_acceleration_structures(
        &self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHRBuilder],
        build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ) -> Result<()> {
        #[allow(clippy::as_conversions)]
        let build_range_infos = build_range_infos
            .iter()
            .map(|r| r as *const vk::AccelerationStructureBuildRangeInfoKHR)
            .collect::<Vec<*const vk::AccelerationStructureBuildRangeInfoKHR>>();

        self.context
            .device
            .build_acceleration_structures_khr(Some(self.raw), infos, &build_range_infos)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!(
                    "Unable to build an acceleration structure on the host: {}",
                    err
                );
                AscheError::VkResult(err)
            })
    }

    /// Copy an acceleration structure on the host.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCopyAccelerationStructureKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn copy_acceleration_structure(
        &self,
        info: &vk::CopyAccelerationStructureInfoKHRBuilder,
    ) -> Result<()> {
        self.context
            .device
            .copy_acceleration_structure_khr(Some(self.raw), info)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!(
                    "Unable to copy an acceleration structure on the host: {}",
                    err
                );
                AscheError::VkResult(err)
            })
    }

    /// Serialize an acceleration structure on the host.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCopyAccelerationStructureToMemoryKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn copy_acceleration_structure_to_memory(
        &self,
        info: &vk::CopyAccelerationStructureToMemoryInfoKHR,
    ) -> Result<()> {
        self.context
            .device
            .copy_acceleration_structure_to_memory_khr(Some(self.raw), info)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!(
                    "Unable to serialize an acceleration structure on the host: {}",
                    err
                );
                AscheError::VkResult(err)
            })
    }

    /// Deserialize an acceleration structure on the host.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCopyMemoryToAccelerationStructureKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn copy_memory_to_acceleration_structure(
        &self,
        info: &vk::CopyMemoryToAccelerationStructureInfoKHR,
    ) -> Result<()> {
        self.context
            .device
            .copy_memory_to_acceleration_structure_khr(Some(self.raw), info)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!(
                    "Unable to deserialize an acceleration structure on the host: {}",
                    err
                );
                AscheError::VkResult(err)
            })
    }
}
