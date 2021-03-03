#![warn(missing_docs)]
//! Provides an abstraction layer above ash to easier use Vulkan in Rust.

use std::sync::Arc;

use erupt::vk;

pub use {
    command::{
        ComputeCommandBuffer, ComputeCommandEncoder, ComputeCommandPool, GraphicsCommandBuffer,
        GraphicsCommandEncoder, GraphicsCommandPool, RenderPassEncoder, TransferCommandBuffer,
        TransferCommandEncoder, TransferCommandPool,
    },
    context::Context,
    descriptor::{DescriptorPool, DescriptorSet, DescriptorSetLayout},
    device::{BARSupport, Device, DeviceConfiguration, QueuePriorityDescriptor},
    error::AscheError,
    instance::{Instance, InstanceConfiguration},
    queue::{ComputeQueue, GraphicsQueue, TransferQueue},
    semaphore::TimelineSemaphore,
    swapchain::SwapchainFrame,
};

pub(crate) mod command;
pub(crate) mod context;
pub(crate) mod descriptor;
pub(crate) mod device;
pub(crate) mod error;
pub(crate) mod instance;
pub(crate) mod queue;
pub(crate) mod semaphore;
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
    /// The raw vk::RenderPass
    pub raw: vk::RenderPass,
    context: Arc<Context>,
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
pub struct PipelineLayout {
    /// The raw vk::PipelineLayout
    pub raw: vk::PipelineLayout,
    context: Arc<Context>,
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
pub struct GraphicsPipeline {
    /// The raw vk::Pipeline.
    pub raw: vk::Pipeline,
    context: Arc<Context>,
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_pipeline(Some(self.raw), None);
        };
    }
}

/// Wraps a compute pipeline.
pub struct ComputePipeline {
    /// The raw vk::Pipeline.
    pub raw: vk::Pipeline,
    context: Arc<Context>,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_pipeline(Some(self.raw), None);
        };
    }
}

/// Wraps a raytracing pipeline.
pub struct RayTracingPipeline {
    /// The raw vk::Pipeline.
    pub raw: vk::Pipeline,
    context: Arc<Context>,
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_pipeline(Some(self.raw), None);
        };
    }
}

/// Wraps a shader module.
pub struct ShaderModule {
    /// The raw vk::ShaderModule.
    pub raw: vk::ShaderModule,
    context: Arc<Context>,
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
    pub samples: vk::SampleCountFlagBits,
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

/// Describes how a sampler should be configured.
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
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            mip_lod_bias: vk::SamplerMipmapMode::LINEAR,
            anisotropy_enable: false,
            max_anisotropy: 0.0,
            compare_op: Some(vk::CompareOp::LESS_OR_EQUAL),
            min_lod: 0.0,
            max_lod: std::f32::MAX,
            border_color: None,
            unnormalized_coordinates: false,
            flags: None,
        }
    }
}

/// Describes how an image view should be configured.
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

/// Describes how a descriptor set should be updated.
pub struct UpdateDescriptorSetDescriptor<'a> {
    /// Binding slot.
    pub binding: u32,
    /// Describes what is updated.
    pub update: DescriptorSetUpdate<'a>,
}

/// That is updated of the descriptor set.
pub enum DescriptorSetUpdate<'a> {
    /// Updates a sampler binding.
    Sampler {
        /// The sampler.
        sampler: &'a Sampler,
    },
    /// Updates a combined image sampler binding.
    CombinedImageSampler {
        /// The sampler.
        sampler: &'a Sampler,
        /// The image view.
        image_view: &'a ImageView,
        /// The layout.
        image_layout: vk::ImageLayout,
    },
    /// Updates a sampled image binding.
    SampledImage {
        /// The image view.
        image_view: &'a ImageView,
    },
    /// Updates a storage image binding.
    StorageImage {
        /// The image view.
        image_view: &'a ImageView,
    },
    /// Updates a uniform texel buffer.
    UniformTexelBuffer {
        /// The texel buffer.
        texel_buffer: &'a BufferView,
    },
    /// Updates a storage texel buffer.
    StorageTexelBuffer {
        /// The texel buffer.
        texel_buffer: &'a BufferView,
    },
    /// Updates an uniform buffer.
    UniformBuffer {
        /// The buffer.
        buffer: &'a Buffer,
        /// Offset inside the buffer.
        offset: u64,
        /// The range inside the buffer.
        range: u64,
    },
    /// Updates a storage buffer.
    StorageBuffer {
        /// The buffer.
        buffer: &'a Buffer,
        /// Offset inside the buffer.
        offset: u64,
        /// The range inside the buffer.
        range: u64,
    },
    /// Updates a dynamic uniform buffer.
    UniformBufferDynamic {
        /// The buffer.
        buffer: &'a Buffer,
        /// Offset inside the buffer.
        offset: u64,
        /// The range inside the buffer.
        range: u64,
    },
    /// Updates a dynamic storage buffer.
    StorageBufferDynamic {
        /// The buffer.
        buffer: &'a Buffer,
        /// Offset inside the buffer.
        offset: u64,
        /// The range inside the buffer.
        range: u64,
    },
    /// Updates an input attachment.
    InputAttachment {
        /// The image view.
        image_view: &'a ImageView,
    },
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
                .deallocate(&self.context.device, &mut self.allocation)
                .expect("can't free buffer allocation");
        };
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

/// Wraps an image.
pub struct Image {
    /// The raw Vulkan image.
    pub raw: vk::Image,
    /// The raw allocation.
    pub allocation: vk_alloc::Allocation,
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

/// Wraps an image view.
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

/// Wraps a sampler.
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
