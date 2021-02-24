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
    device::{Device, DeviceConfiguration, QueuePriorityDescriptor},
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
    pub(crate) context: Arc<Context>,
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
    pub(crate) context: Arc<Context>,
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
    pub(crate) context: Arc<Context>,
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
    pub(crate) context: Arc<Context>,
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
    pub(crate) context: Arc<Context>,
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

/// Wraps a buffer.
pub struct Buffer {
    pub(crate) context: Arc<Context>,
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
