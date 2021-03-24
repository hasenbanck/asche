use std::sync::Arc;

use erupt::vk;

use crate::Context;

/// Wraps an image.
#[derive(Debug)]
pub struct Image {
    /// The raw Vulkan image.
    pub raw: vk::Image,
    allocation: vk_alloc::Allocation,
    context: Arc<Context>,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_image(Some(self.raw), None);
            self.context
                .allocator
                .lock()
                .deallocate(&self.context.device, &self.allocation)
                .expect("can't free image allocation");
        };
    }
}

impl Image {
    pub(crate) fn new(
        raw: vk::Image,
        allocation: vk_alloc::Allocation,
        context: Arc<Context>,
    ) -> Self {
        Self {
            raw,
            allocation,
            context,
        }
    }
}

/// Wraps an image view.
#[derive(Debug)]
pub struct ImageView {
    /// The raw Vulkan image view.
    pub raw: vk::ImageView,
    context: Arc<Context>,
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_image_view(Some(self.raw), None);
        };
    }
}

impl ImageView {
    pub(crate) fn new(raw: vk::ImageView, context: Arc<Context>) -> Self {
        Self { raw, context }
    }
}

/// Wraps a sampler.
#[derive(Debug)]
pub struct Sampler {
    /// The raw Vulkan sampler.
    pub raw: vk::Sampler,
    context: Arc<Context>,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_sampler(Some(self.raw), None);
        };
    }
}

impl Sampler {
    pub(crate) fn new(raw: vk::Sampler, context: Arc<Context>) -> Self {
        Self { raw, context }
    }
}
