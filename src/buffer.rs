use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing1::error;

use crate::{context::Context, memory_allocator::MemoryAllocator, AscheError, Lifetime, Result};

/// Wraps a buffer.
#[derive(Debug)]
pub struct Buffer<LT: Lifetime> {
    raw: vk::Buffer,
    allocation: vk_alloc::Allocation<LT>,
    memory_allocator: Arc<MemoryAllocator<LT>>,
    context: Arc<Context>,
}

impl<LT: Lifetime> Drop for Buffer<LT> {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_buffer(Some(self.raw), None);
            self.memory_allocator
                .allocator
                .deallocate(&self.context.device, &self.allocation)
                .expect("can't free buffer allocation");
        };
    }
}

impl<LT: Lifetime> Buffer<LT> {
    pub(crate) fn new(
        raw: vk::Buffer,
        allocation: vk_alloc::Allocation<LT>,
        memory_allocator: Arc<MemoryAllocator<LT>>,
        context: Arc<Context>,
    ) -> Self {
        Self {
            raw,
            allocation,
            memory_allocator,
            context,
        }
    }

    /// The raw Vulkan buffer handle.
    #[inline]
    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }

    /// Returns a valid mapped slice if the buffer memory is host visible, otherwise it will return None.
    #[inline]
    pub unsafe fn mapped_slice(&self) -> Result<Option<&[u8]>> {
        let slice = self.allocation.mapped_slice()?;
        Ok(slice)
    }

    /// Returns a valid mapped mutable slice if the buffer memory is host visible, otherwise it will return None.
    #[inline]
    pub unsafe fn mapped_slice_mut(&mut self) -> Result<Option<&mut [u8]>> {
        let slice = self.allocation.mapped_slice_mut()?;
        Ok(slice)
    }

    /// Flush the mapped memory of the buffer. Used for CPU->GPU transfers.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkFlushMappedMemoryRanges.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn flush(&self) -> Result<()> {
        let ranges = [vk::MappedMemoryRangeBuilder::new()
            .memory(self.allocation.device_memory())
            .size(self.allocation.size())
            .offset(self.allocation.offset())];
        self.context
            .device
            .flush_mapped_memory_ranges(&ranges)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to flush a mapped memory range: {}", err);
                AscheError::VkResult(err)
            })?;

        Ok(())
    }

    /// Invalidate the mapped memory of the buffer. Used for GPU->CPU transfers.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkInvalidateMappedMemoryRanges.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn invalidate(&self) -> Result<()> {
        let ranges = [vk::MappedMemoryRangeBuilder::new()
            .memory(self.allocation.device_memory())
            .size(self.allocation.size())
            .offset(self.allocation.offset())];
        self.context
            .device
            .invalidate_mapped_memory_ranges(&ranges)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to invalidate a mapped memory range: {}", err);
                AscheError::VkResult(err)
            })?;

        Ok(())
    }

    /// Query an address of a buffer.
    #[cfg(feature = "vk-buffer-device-address")]
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetBufferDeviceAddress.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn device_address(&self) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfoBuilder::new().buffer(self.raw);
        self.context.device.get_buffer_device_address(&info)
    }
}

/// Wraps a buffer view.
#[derive(Debug)]
pub struct BufferView {
    raw: vk::BufferView,
    context: Arc<Context>,
}

impl Drop for BufferView {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_buffer_view(Some(self.raw), None);
        };
    }
}

impl BufferView {
    pub(crate) fn new(raw: vk::BufferView, context: Arc<Context>) -> Self {
        Self { raw, context }
    }

    /// The raw Vulkan buffer view handle.
    #[inline]
    pub fn raw(&self) -> vk::BufferView {
        self.raw
    }
}
