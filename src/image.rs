use std::sync::Arc;

use erupt::vk;

use crate::context::Context;

/// Wraps an image.
#[derive(Debug)]
pub struct Image {
    raw: vk::Image,
    allocation: vk_alloc::Allocation,
    context: Arc<Context>,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_image(Some(self.raw), None);
            self.context
                .allocator
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

    /// The raw Vulkan image handle.
    #[inline]
    pub fn raw(&self) -> vk::Image {
        self.raw
    }
}

/// Wraps an image view.
#[derive(Debug)]
pub struct ImageView {
    raw: vk::ImageView,
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

    /// The raw Vulkan image view handle.
    #[inline]
    pub fn raw(&self) -> vk::ImageView {
        self.raw
    }
}

/// Wraps a sampler.
#[derive(Debug)]
pub struct Sampler {
    raw: vk::Sampler,
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

    /// The raw Vulkan sampler handle.
    #[inline]
    pub fn raw(&self) -> vk::Sampler {
        self.raw
    }
}
