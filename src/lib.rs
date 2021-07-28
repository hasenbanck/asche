#![warn(missing_docs)]
#![deny(clippy::as_conversions)]
#![deny(clippy::panic)]
#![deny(clippy::unwrap_used)]

//! Provides an abstraction layer above erupt to easier use Vulkan in Rust.
//!
//! No validation and a lot of pain. Lifetimes are not fully tracked, so you need to pay attention
//! when to drop which resource to avoid UB (check the validation layer). In general resources
//! should not be dropped while they are being used.

use std::sync::Arc;

use erupt::vk;

pub use {
    acceleration_structure::AccelerationStructure,
    buffer::{Buffer, BufferView},
    command::{
        CommandBufferSemaphore, CommonCommands, ComputeCommandBuffer, ComputeCommandEncoder,
        ComputeCommandPool, GraphicsCommandBuffer, GraphicsCommandEncoder, GraphicsCommandPool,
        RenderPassEncoder, TransferCommandBuffer, TransferCommandEncoder, TransferCommandPool,
    },
    deferred_operation::DeferredOperation,
    descriptor::{DescriptorPool, DescriptorSet, DescriptorSetLayout},
    device::{BarSupport, Device, DeviceConfiguration, QueueConfiguration, Queues},
    error::AscheError,
    fence::Fence,
    image::{Image, ImageView, Sampler},
    instance::{Instance, InstanceConfiguration, Version},
    query::QueryPool,
    queue::{ComputeQueue, GraphicsQueue, TransferQueue},
    semaphore::{
        BinarySemaphore, BinarySemaphoreHandle, TimelineSemaphore, TimelineSemaphoreHandle,
    },
    swapchain::{Swapchain, SwapchainFrame},
    vk_alloc::{AllocatorError, Lifetime, MemoryLocation},
};

use crate::context::Context;

mod acceleration_structure;
mod buffer;
mod command;
mod context;
mod deferred_operation;
mod descriptor;
mod device;
mod error;
mod fence;
mod image;
mod instance;
mod memory_allocator;
mod query;
mod queue;
mod semaphore;
mod swapchain;
#[cfg(debug_assertions)]
mod vk_debug;

type Result<T> = std::result::Result<T, AscheError>;

/// Type of a queue.
#[derive(Copy, Clone, Debug)]
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
#[derive(Debug)]
pub struct RenderPass {
    raw: vk::RenderPass,
    context: Arc<Context>,
}

impl RenderPass {
    /// The raw Vulkan render pass handle.
    #[inline]
    pub fn raw(&self) -> vk::RenderPass {
        self.raw
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_render_pass(Some(self.raw), None);
        };
    }
}

/// Wraps a pipeline layout.
#[derive(Debug)]
pub struct PipelineLayout {
    raw: vk::PipelineLayout,
    context: Arc<Context>,
}

impl PipelineLayout {
    /// The raw Vulkan pipeline layout handle.
    #[inline]
    pub fn raw(&self) -> vk::PipelineLayout {
        self.raw
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_pipeline_layout(Some(self.raw), None);
        };
    }
}

/// Wraps a graphics pipeline.
#[derive(Debug)]
pub struct GraphicsPipeline {
    raw: vk::Pipeline,
    context: Arc<Context>,
}

impl GraphicsPipeline {
    /// The raw Vulkan pipeline handle.
    #[inline]
    pub fn raw(&self) -> vk::Pipeline {
        self.raw
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_pipeline(Some(self.raw), None);
        };
    }
}

/// Wraps a compute pipeline.
#[derive(Debug)]
pub struct ComputePipeline {
    raw: vk::Pipeline,
    context: Arc<Context>,
}

impl ComputePipeline {
    /// The raw Vulkan pipeline handle.
    #[inline]
    pub fn raw(&self) -> vk::Pipeline {
        self.raw
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_pipeline(Some(self.raw), None);
        };
    }
}

/// Wraps a raytracing pipeline.
#[derive(Debug)]
pub struct RayTracingPipeline {
    raw: vk::Pipeline,
    context: Arc<Context>,
}

impl RayTracingPipeline {
    /// The raw Vulkan pipeline handle.
    #[inline]
    pub fn raw(&self) -> vk::Pipeline {
        self.raw
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_pipeline(Some(self.raw), None);
        };
    }
}

/// Wraps a shader module.
#[derive(Debug)]
pub struct ShaderModule {
    raw: vk::ShaderModule,
    context: Arc<Context>,
}

impl ShaderModule {
    /// The raw Vulkan shader module handle.
    #[inline]
    pub fn raw(&self) -> vk::ShaderModule {
        self.raw
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_shader_module(Some(self.raw), None);
        };
    }
}

/// Describes how an image should be configured.
#[derive(Clone, Debug)]
pub struct BufferDescriptor<'a, LT: Lifetime> {
    /// Name used for debugging.
    pub name: &'a str,
    /// What is the buffer used for.
    pub usage: vk::BufferUsageFlags,
    /// Where should the buffer reside.
    pub memory_location: vk_alloc::MemoryLocation,
    /// The lifetime of an allocation. Used to pool allocations and reduce fragmentation.
    pub lifetime: LT,
    /// The sharing mode between queues.
    pub sharing_mode: vk::SharingMode,
    /// Which queues should have access to it.
    pub queues: vk::QueueFlags,
    /// The size of the buffer.
    pub size: vk::DeviceSize,
    /// Additional flags.
    pub flags: Option<vk::BufferCreateFlags>,
}

/// Describes how an buffer view should be configured.
#[derive(Clone, Debug)]
pub struct BufferViewDescriptor<'a, LT: Lifetime> {
    /// Name used for debugging.
    pub name: &'a str,
    /// The handle of the buffer.
    pub buffer: &'a Buffer<LT>,
    /// The format of the buffer view.
    pub format: vk::Format,
    /// Offset.
    pub offset: vk::DeviceSize,
    /// Range.
    pub range: u64,
    /// Additional flags.
    pub flags: Option<vk::BufferViewCreateFlags>,
}

/// Describes how an image should be configured.
#[derive(Clone, Debug)]
pub struct ImageDescriptor<'a, LT: Lifetime> {
    /// Name used for debugging.
    pub name: &'a str,
    /// What is the image used for.
    pub usage: vk::ImageUsageFlags,
    /// Where should the image reside.
    pub memory_location: vk_alloc::MemoryLocation,
    /// The lifetime of an allocation. Used to pool allocations and reduce fragmentation.
    pub lifetime: LT,
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
    pub samples: vk::SampleCountFlagBits,
    /// The tiling used.
    pub tiling: vk::ImageTiling,
    /// The initial format.
    pub initial_layout: vk::ImageLayout,
    /// Additional flags.
    pub flags: Option<vk::ImageCreateFlags>,
}

/// Describes how an image view should be configured.
#[derive(Clone, Debug)]
pub struct ImageViewDescriptor<'a, LT: Lifetime> {
    /// Name used for debugging.
    pub name: &'a str,
    /// The handle of the image.
    pub image: &'a Image<LT>,
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

/// Describes how a sampler should be configured.
#[derive(Clone, Debug)]
pub struct SamplerDescriptor<'a> {
    /// Name used for debugging.
    pub name: &'a str,
    /// Filter used for magnification.
    pub mag_filter: vk::Filter,
    /// Filter used for minification.
    pub min_filter: vk::Filter,
    /// Mipmap mode.
    pub mipmap_mode: vk::SamplerMipmapMode,
    /// Address mode U.
    pub address_mode_u: vk::SamplerAddressMode,
    /// Address mode V.
    pub address_mode_v: vk::SamplerAddressMode,
    /// Address mode W.
    pub address_mode_w: vk::SamplerAddressMode,
    /// Mipmap load bias.
    pub mip_lod_bias: vk::SamplerMipmapMode,
    /// Anisotropy filtering enabled.
    pub anisotropy_enable: bool,
    /// The anisotropy filter rate.
    pub max_anisotropy: f32,
    /// Optional Compare operation.
    pub compare_op: Option<vk::CompareOp>,
    /// Minimal LOD.
    pub min_lod: f32,
    /// Maximal LOD.
    pub max_lod: f32,
    /// Border color.
    pub border_color: Option<vk::BorderColor>,
    /// Un-normalized coordinates.
    pub unnormalized_coordinates: bool,
    /// Optional flags.
    pub flags: Option<vk::SamplerCreateFlags>,
}

impl<'a> Default for SamplerDescriptor<'a> {
    fn default() -> Self {
        Self {
            name: "",
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            mip_lod_bias: vk::SamplerMipmapMode::NEAREST,
            anisotropy_enable: false,
            max_anisotropy: 0.0,
            compare_op: None,
            min_lod: 0.0,
            max_lod: f32::MAX,
            border_color: None,
            unnormalized_coordinates: false,
            flags: None,
        }
    }
}

/// Describes how an image view should be configured.
#[derive(Clone, Debug)]
pub struct DescriptorPoolDescriptor<'a> {
    /// Name used for debugging.
    pub name: &'a str,
    /// Max sets.
    pub max_sets: u32,
    /// All sizes of the pool.
    pub pool_sizes: &'a [vk::DescriptorPoolSizeBuilder<'a>],
    /// Optional flags.
    pub flags: Option<vk::DescriptorPoolCreateFlags>,
}

/// Describes a render pass color attachment. Used to create the framebuffer.
#[derive(Clone, Debug)]
pub struct RenderPassColorAttachmentDescriptor {
    /// The Vulkan image view of the attachment.
    pub attachment: vk::ImageView,
    /// Value used to clear the attachment.
    pub clear_value: Option<vk::ClearValue>,
}

/// Describes a render pass depth attachment. Used to create the framebuffer.
#[derive(Clone, Debug)]
pub struct RenderPassDepthAttachmentDescriptor {
    /// The Vulkan image view of the attachment.
    pub attachment: vk::ImageView,
    /// Value used to clear the attachment.
    pub clear_value: Option<vk::ClearValue>,
}
