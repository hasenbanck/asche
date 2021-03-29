use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "smallvec")]
use smallvec::SmallVec;
#[cfg(feature = "tracing")]
use tracing::error;

use crate::context::Context;
use crate::{AscheError, Result};

/// Wraps a deferred operation.
#[derive(Debug)]
pub struct DeferredOperation {
    /// The raw Vulkan deferred operation.
    pub raw: vk::DeferredOperationKHR,
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

    /// Assign a thread to a deferred operation.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDeferredOperationJoinKHR.html
    pub fn join(&self) -> Result<()> {
        unsafe { self.context.device.deferred_operation_join_khr(self.raw) }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to assign a thread to a deferred operation: {}", err);
            AscheError::VkResult(err)
        })
    }

    /// Query the maximum concurrency on a deferred operation.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetDeferredOperationMaxConcurrencyKHR.html
    pub fn max_concurrency(&self) -> u32 {
        unsafe {
            self.context
                .device
                .get_deferred_operation_max_concurrency_khr(self.raw)
        }
    }

    /// Query the result of a deferred operation.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetDeferredOperationResultKHR.html
    pub fn result(&self) -> Result<()> {
        unsafe {
            self.context
                .device
                .get_deferred_operation_result_khr(self.raw)
        }
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
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkBuildAccelerationStructuresKHR.html
    pub fn build_acceleration_structures(
        &self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHRBuilder],
        build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ) -> Result<()> {
        let build_range_infos = build_range_infos
            .iter()
            .map(|r| r as *const vk::AccelerationStructureBuildRangeInfoKHR);

        #[cfg(feature = "smallvec")]
        let build_range_infos = build_range_infos
            .collect::<SmallVec<[*const vk::AccelerationStructureBuildRangeInfoKHR; 4]>>();

        #[cfg(not(feature = "smallvec"))]
        let build_range_infos =
            build_range_infos.collect::<Vec<*const vk::AccelerationStructureBuildRangeInfoKHR>>();

        unsafe {
            self.context.device.build_acceleration_structures_khr(
                Some(self.raw),
                infos,
                &build_range_infos,
            )
        }
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
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCopyAccelerationStructureKHR.html
    pub fn copy_acceleration_structure(
        &self,
        info: &vk::CopyAccelerationStructureInfoKHRBuilder,
    ) -> Result<()> {
        unsafe {
            self.context
                .device
                .copy_acceleration_structure_khr(Some(self.raw), info)
        }
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
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCopyAccelerationStructureToMemoryKHR.html
    pub fn copy_acceleration_structure_to_memory(
        &self,
        info: &vk::CopyAccelerationStructureToMemoryInfoKHR,
    ) -> Result<()> {
        unsafe {
            self.context
                .device
                .copy_acceleration_structure_to_memory_khr(Some(self.raw), info)
        }
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
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCopyMemoryToAccelerationStructureKHR.html
    pub fn copy_memory_to_acceleration_structure(
        &self,
        info: &vk::CopyMemoryToAccelerationStructureInfoKHR,
    ) -> Result<()> {
        unsafe {
            self.context
                .device
                .copy_memory_to_acceleration_structure_khr(Some(self.raw), info)
        }
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
