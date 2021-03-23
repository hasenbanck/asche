use std::sync::Arc;

use erupt::vk;

use crate::Context;

/// Wraps a buffer.
pub struct Buffer {
    /// The raw Vulkan buffer.
    pub raw: vk::Buffer,
    /// The raw allocation.
    pub allocation: vk_alloc::Allocation,
    context: Arc<Context>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_buffer(Some(self.raw), None);
            self.context
                .allocator
                .lock()
                .unwrap()
                .deallocate(&self.context.device, &self.allocation)
                .expect("can't free buffer allocation");
        };
    }
}

impl Buffer {
    pub(crate) fn new(
        raw: vk::Buffer,
        allocation: vk_alloc::Allocation,
        context: Arc<Context>,
    ) -> Self {
        Self {
            raw,
            allocation,
            context,
        }
    }

    /// Query an address of a buffer.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetBufferDeviceAddress.html
    pub fn device_address(&self) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfoBuilder::new().buffer(self.raw);
        unsafe { self.context.device.get_buffer_device_address(&info) }
    }
}

/// Wraps a buffer view.
pub struct BufferView {
    /// The raw Vulkan buffer view.
    pub raw: vk::BufferView,
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
}
