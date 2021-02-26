#![warn(missing_docs)]
//! Provides an abstraction layer above ash to easier use Vulkan in Rust.

use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;

pub use {
    command::{
        ComputeCommandBuffer, ComputeCommandEncoder, ComputeCommandPool, GraphicsCommandBuffer,
        GraphicsCommandEncoder, GraphicsCommandPool, RenderPassEncoder, TransferCommandBuffer,
        TransferCommandEncoder, TransferCommandPool,
    },
    context::Context,
    device::{BARSupport, Device, DeviceConfiguration, QueuePriorityDescriptor},
    error::AscheError,
    instance::{Instance, InstanceConfiguration},
    queue::{ComputeQueue, GraphicsQueue, TransferQueue},
    swapchain::SwapchainFrame,
};

pub(crate) mod command;
pub(crate) mod context;
pub(crate) mod device;
pub(crate) mod error;
pub(crate) mod instance;
pub(crate) mod queue;
pub(crate) mod swapchain;
#[cfg(debug_assertions)]
pub(crate) mod vk_debug;

pub(crate) type Result<T> = std::result::Result<T, AscheError>;

/// Type of a queue.
#[derive(Copy, Clone)]
pub(crate) enum QueueType {
    /// Compute queue.
    Compute = 0,
    /// Graphics queue.
    Graphics,
    /// Transfer queue.
    Transfer,
}

impl std::fmt::Display for QueueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueueType::Compute => f.write_str("Compute"),
            QueueType::Graphics => f.write_str("Graphics"),
            QueueType::Transfer => f.write_str("Transfer"),
        }
    }
}

/// Wraps a render pass.
pub struct RenderPass {
    context: Arc<Context>,
    /// The raw vk::RenderPass
    pub raw: vk::RenderPass,
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_render_pass(self.raw, None);
        };
    }
}

/// Wraps a pipeline layout.
pub struct PipelineLayout {
    context: Arc<Context>,
    /// The raw vk::PipelineLayout
    pub raw: vk::PipelineLayout,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_pipeline_layout(self.raw, None);
        };
    }
}

/// Wraps a graphics pipeline.
pub struct GraphicsPipeline {
    context: Arc<Context>,
    /// The raw vk::Pipeline.
    pub raw: vk::Pipeline,
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.logical_device.destroy_pipeline(self.raw, None);
        };
    }
}

/// Wraps a compute pipeline.
pub struct ComputePipeline {
    context: Arc<Context>,
    /// The raw vk::Pipeline.
    pub raw: vk::Pipeline,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.logical_device.destroy_pipeline(self.raw, None);
        };
    }
}

/// Wraps a shader module.
pub struct ShaderModule {
    context: Arc<Context>,
    /// The raw vk::ShaderModule.
    pub raw: vk::ShaderModule,
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_shader_module(self.raw, None);
        };
    }
}

/// Describes how an image should be configured.
pub struct BufferDescriptor<'a> {
    /// Name used for debugging.
    pub name: &'a str,
    /// What is the buffer used for.
    pub usage: vk::BufferUsageFlags,
    /// Where should the buffer reside.
    pub memory_location: vk_alloc::MemoryLocation,
    /// The sharing mode between queues.
    pub sharing_mode: vk::SharingMode,
    /// Which queues should have access to it.
    pub queues: vk::QueueFlags,
    /// The size of the buffer.
    pub size: u64,
    /// Additional flags.
    pub flags: Option<vk::BufferCreateFlags>,
}

/// Describes how an image should be configured.
pub struct ImageDescriptor<'a> {
    /// Name used for debugging.
    pub name: &'a str,
    /// What is the image used for.
    pub usage: vk::ImageUsageFlags,
    /// Where should the image reside.
    pub memory_location: vk_alloc::MemoryLocation,
    /// The sharing mode between queues.
    pub sharing_mode: vk::SharingMode,
    /// Which queues should have access to it.
    pub queues: vk::QueueFlags,
    /// The type of the image.
    pub image_type: vk::ImageType,
    /// The format of the image.
    pub format: vk::Format,
    /// The extent of the image.
    pub extent: vk::Extent3D,
    /// The count mips level.
    pub mip_levels: u32,
    /// The count array layers.
    pub array_layers: u32,
    /// Sample count flags.
    pub samples: vk::SampleCountFlags,
    /// The tiling used.
    pub tiling: vk::ImageTiling,
    /// The initial format.
    pub initial_layout: vk::ImageLayout,
    /// Additional flags.
    pub flags: Option<vk::ImageCreateFlags>,
}

/// Describes how an image view should be configured.
pub struct ImageViewDescriptor<'a> {
    /// Name used for debugging.
    pub name: &'a str,
    /// The handle of the image.
    pub image: &'a Image,
    /// The type of the image view.
    pub view_type: vk::ImageViewType,
    /// The format of the image view.
    pub format: vk::Format,
    /// Component mapping.
    pub components: vk::ComponentMapping,
    /// The subresource range.
    pub subresource_range: vk::ImageSubresourceRange,
    /// Additional flags.
    pub flags: Option<vk::ImageViewCreateFlags>,
}

/// Describes a render pass color attachment. Used to create the framebuffer.
pub struct RenderPassColorAttachmentDescriptor {
    /// The Vulkan image view of the attachment.
    pub attachment: vk::ImageView,
    /// Value used to clear the attachment.
    pub clear_value: vk::ClearValue,
}

/// Describes a render pass depth attachment. Used to create the framebuffer.
pub struct RenderPassDepthAttachmentDescriptor {
    /// The Vulkan image view of the attachment.
    pub attachment: vk::ImageView,
    /// Value used to clear the attachment.
    pub clear_value: vk::ClearValue,
}

/// Wraps a buffer.
pub struct Buffer {
    context: Arc<Context>,
    /// The raw allocation.
    pub allocation: vk_alloc::Allocation,
    /// The raw Vulkan buffer.
    pub raw: vk::Buffer,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .allocator
                .lock()
                .free(&self.allocation)
                .expect("can't free buffer allocation");
            self.context.logical_device.destroy_buffer(self.raw, None);
        };
    }
}

/// Wraps an image.
pub struct Image {
    context: Arc<Context>,
    /// The raw allocation.
    pub allocation: vk_alloc::Allocation,
    /// The raw Vulkan image.
    pub raw: vk::Image,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.context
                .allocator
                .lock()
                .free(&self.allocation)
                .expect("can't free image allocation");
            self.context.logical_device.destroy_image(self.raw, None);
        };
    }
}

/// Wraps an image view.
pub struct ImageView {
    context: Arc<Context>,
    /// The raw Vulkan image view.
    pub raw: vk::ImageView,
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_image_view(self.raw, None);
        };
    }
}

/// A wrapper timeline semaphore.
pub struct TimelineSemaphore {
    context: Arc<Context>,
    pub(crate) raw: vk::Semaphore,
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_semaphore(self.raw, None);
        };
    }
}
