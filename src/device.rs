use std::os::raw::c_char;
use std::sync::Arc;

use erupt::{vk, ExtendableFrom};
#[cfg(feature = "tracing")]
use tracing::info;

use crate::context::Context;
use crate::instance::Instance;
use crate::semaphore::TimelineSemaphore;
use crate::swapchain::{Swapchain, SwapchainDescriptor, SwapchainFrame};
use crate::{
    AscheError, Buffer, BufferDescriptor, ComputePipeline, ComputeQueue, DescriptorPool,
    DescriptorPoolDescriptor, DescriptorSetLayout, GraphicsPipeline, GraphicsQueue, Image,
    ImageDescriptor, ImageView, ImageViewDescriptor, PipelineLayout, RenderPass, Result, Sampler,
    SamplerDescriptor, ShaderModule, TransferQueue,
};

/// Defines the priorities of the queues.
pub struct QueuePriorityDescriptor {
    /// Priority of the graphics queue.
    pub graphics: f32,
    /// Priority of the transfer queue.
    pub transfer: f32,
    /// Priority of the compute queue.
    pub compute: f32,
}

/// Describes how the device should be configured.
pub struct DeviceConfiguration<'a> {
    /// The device type that is requested.
    pub device_type: vk::PhysicalDeviceType,
    /// The image format of the swapchain.
    pub swapchain_format: vk::Format,
    /// The color space of the swapchain.
    pub swapchain_color_space: vk::ColorSpaceKHR,
    /// The presentation mode of the swap chain.
    pub presentation_mode: vk::PresentModeKHR,
    /// The priorities of the queues.
    pub queue_priority: QueuePriorityDescriptor,
    /// Device extensions to load.
    pub extensions: Vec<*const c_char>,
    /// Vulkan 1.0 features.
    pub features_v1_0: Option<vk::PhysicalDeviceFeaturesBuilder<'a>>,
    /// Vulkan 1.0 features.
    pub features_v1_1: Option<vk::PhysicalDeviceVulkan11FeaturesBuilder<'a>>,
    /// Vulkan 1.1 features
    pub features_v1_2: Option<vk::PhysicalDeviceVulkan12FeaturesBuilder<'a>>,
}

impl<'a> Default for DeviceConfiguration<'a> {
    fn default() -> Self {
        Self {
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
            swapchain_format: vk::Format::B8G8R8A8_SRGB,
            swapchain_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR,
            presentation_mode: vk::PresentModeKHR::FIFO_KHR,
            queue_priority: QueuePriorityDescriptor {
                graphics: 1.0,
                transfer: 1.0,
                compute: 1.0,
            },
            extensions: vec![],
            features_v1_0: None,
            features_v1_1: None,
            features_v1_2: None,
        }
    }
}

/// Shows if the device support access to the device memory using the base address register.
pub enum BARSupport {
    /// Device doesn't support BAR.
    NotSupported,
    /// Device supports BAR.
    BAR,
    /// Device supports resizable BAR.
    ResizableBAR,
}

impl std::fmt::Display for BARSupport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BARSupport::NotSupported => f.write_str("No BAR support"),
            BARSupport::BAR => f.write_str("BAR supported"),
            BARSupport::ResizableBAR => f.write_str("Resizable BAR supported"),
        }
    }
}

/// A Vulkan device.
///
/// Handles all resource creation, swapchain and framebuffer handling. Command buffer and queue handling are handled by the `Queue`.
pub struct Device {
    compute_queue_family_index: u32,
    graphic_queue_family_index: u32,
    transfer_queue_family_index: u32,
    swapchain: Option<Swapchain>,
    swapchain_format: vk::Format,
    swapchain_color_space: vk::ColorSpaceKHR,
    presentation_mode: vk::PresentModeKHR,
    /// The type of the physical device.
    pub device_type: vk::PhysicalDeviceType,
    /// Shows if the device support access to the device memory using the base address register.
    pub resizable_bar_support: BARSupport,
    context: Arc<Context>,
}

impl Device {
    /// Creates a new device and three queues:
    ///  * Compute queue
    ///  * Graphics queue
    ///  * Transfer queue
    ///
    /// The compute and transfer queue use dedicated queue families if provided by the implementation.
    /// The graphics queue is guaranteed to be able to write the the surface.
    #[allow(unused_variables)]
    pub(crate) fn new(
        instance: Instance,
        configuration: DeviceConfiguration,
    ) -> Result<(Self, (ComputeQueue, GraphicsQueue, TransferQueue))> {
        let (physical_device, physical_device_properties, physical_device_driver_properties) =
            instance.find_physical_device(configuration.device_type)?;

        let resizable_bar_support =
            query_support_resizable_bar(&instance, physical_device, &physical_device_properties);

        #[cfg(feature = "tracing")]
        {
            let name = String::from(
                unsafe {
                    std::ffi::CStr::from_ptr(physical_device_properties.device_name.as_ptr())
                }
                .to_str()?,
            );

            info!(
                "Selected physical device: {} ({:?})",
                name, physical_device_properties.device_type
            );

            let driver_name = String::from(
                unsafe {
                    std::ffi::CStr::from_ptr(physical_device_driver_properties.driver_name.as_ptr())
                }
                .to_str()?,
            );

            let driver_version = String::from(
                unsafe {
                    std::ffi::CStr::from_ptr(physical_device_driver_properties.driver_info.as_ptr())
                }
                .to_str()?,
            );

            info!(
                "Driver version of selected device: {} ({})",
                driver_name, driver_version
            );

            info!("BAR support: {}", resizable_bar_support);
        }

        let presentation_mode = configuration.presentation_mode;
        let swapchain_format = configuration.swapchain_format;
        let swapchain_color_space = configuration.swapchain_color_space;

        #[cfg(feature = "tracing")]
        info!("Creating logical device and queues");

        let (device, family_ids, queues) =
            instance.create_device(physical_device, configuration)?;

        #[cfg(feature = "tracing")]
        info!("Creating Vulkan memory allocator");

        let allocator = vk_alloc::Allocator::new(
            &instance.raw,
            physical_device,
            &vk_alloc::AllocatorDescriptor::default(),
        );

        let context = Arc::new(Context::new(instance, device, physical_device, allocator));

        let compute_queue = ComputeQueue::new(context.clone(), family_ids[0], queues[0]);
        let graphics_queue = GraphicsQueue::new(context.clone(), family_ids[1], queues[1]);
        let transfer_queue = TransferQueue::new(context.clone(), family_ids[2], queues[2]);

        context.set_object_name(
            "Compute Queue",
            vk::ObjectType::QUEUE,
            compute_queue.inner.raw.0 as u64,
        )?;
        context.set_object_name(
            "Graphics Queue",
            vk::ObjectType::QUEUE,
            graphics_queue.inner.raw.0 as u64,
        )?;
        context.set_object_name(
            "Transfer Queue",
            vk::ObjectType::QUEUE,
            transfer_queue.inner.raw.0 as u64,
        )?;

        let mut device = Device {
            device_type: physical_device_properties.device_type,
            resizable_bar_support,
            context,
            compute_queue_family_index: family_ids[0],
            graphic_queue_family_index: family_ids[1],
            transfer_queue_family_index: family_ids[2],
            presentation_mode,
            swapchain_format,
            swapchain_color_space,
            swapchain: None,
        };

        device.recreate_swapchain(None)?;

        Ok((device, (compute_queue, graphics_queue, transfer_queue)))
    }

    /// Returns a reference to the context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Recreates the swapchain. Needs to be called if the surface has changed.
    pub fn recreate_swapchain(&mut self, window_extend: Option<vk::Extent2D>) -> Result<()> {
        self.context.destroy_framebuffer();

        #[cfg(feature = "tracing")]
        info!(
            "Creating swapchain with format {:?} and color space {:?}",
            self.swapchain_format, self.swapchain_color_space
        );

        let formats = unsafe {
            self.context
                .instance
                .raw
                .get_physical_device_surface_formats_khr(
                    self.context.physical_device,
                    self.context.instance.surface,
                    None,
                )
                .result()?
        };

        let capabilities = unsafe {
            self.context
                .instance
                .raw
                .get_physical_device_surface_capabilities_khr(
                    self.context.physical_device,
                    self.context.instance.surface,
                    None,
                )
                .result()?
        };

        let presentation_mode = unsafe {
            self.context
                .instance
                .raw
                .get_physical_device_surface_present_modes_khr(
                    self.context.physical_device,
                    self.context.instance.surface,
                    None,
                )
                .result()?
        };

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let extent = match capabilities.current_extent.width {
            std::u32::MAX => window_extend.unwrap_or_default(),
            _ => capabilities.current_extent,
        };

        let pre_transform = if capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY_KHR)
        {
            vk::SurfaceTransformFlagBitsKHR::IDENTITY_KHR
        } else {
            capabilities.current_transform
        };

        let format = formats
            .iter()
            .find(|f| {
                f.format == self.swapchain_format && f.color_space == self.swapchain_color_space
            })
            .ok_or(AscheError::SwapchainFormatIncompatible)?;

        let presentation_mode = *presentation_mode
            .iter()
            .find(|m| **m == self.presentation_mode)
            .ok_or(AscheError::PresentationModeUnsupported)?;

        let old_swapchain = self.swapchain.take();

        let swapchain = Swapchain::new(
            self.context.clone(),
            SwapchainDescriptor {
                graphic_queue_family_index: self.graphic_queue_family_index,
                extent,
                pre_transform,
                format: format.format,
                color_space: format.color_space,
                presentation_mode,
                image_count,
            },
            old_swapchain,
        )?;

        #[cfg(feature = "tracing")]
        info!("Swapchain has {} image(s)", image_count);

        self.swapchain = Some(swapchain);

        Ok(())
    }

    /// Gets the next frame the program can render into.
    pub fn get_next_frame(&self) -> Result<SwapchainFrame> {
        let swapchain = self
            .swapchain
            .as_ref()
            .ok_or(AscheError::SwapchainNotInitialized)?;
        swapchain.get_next_frame()
    }

    /// Queues the frame in the presentation queue.
    pub fn queue_frame(&self, graphics_queue: &GraphicsQueue, frame: SwapchainFrame) -> Result<()> {
        let swapchain = &self
            .swapchain
            .as_ref()
            .ok_or(AscheError::SwapchainNotInitialized)?;
        swapchain.queue_frame(frame, graphics_queue.inner.raw)
    }

    /// Creates a new render pass.
    pub fn create_render_pass(
        &mut self,
        name: &str,
        renderpass_info: vk::RenderPassCreateInfoBuilder,
    ) -> Result<RenderPass> {
        debug_assert!(
            renderpass_info.attachment_count <= 4,
            "Maximum size of attachments reached. This limit is artificial for the RenderpassEncoder.begin_render_pass() method."
        );

        let renderpass = unsafe {
            self.context
                .device
                .create_render_pass(&renderpass_info, None, None)
                .result()?
        };

        self.context
            .set_object_name(name, vk::ObjectType::RENDER_PASS, renderpass.0)?;

        Ok(RenderPass {
            context: self.context.clone(),
            raw: renderpass,
        })
    }

    /// Creates a new pipeline layout.
    pub fn create_pipeline_layout(
        &mut self,
        name: &str,
        pipeline_layout_info: vk::PipelineLayoutCreateInfoBuilder,
    ) -> Result<PipelineLayout> {
        let pipeline_layout = unsafe {
            self.context
                .device
                .create_pipeline_layout(&pipeline_layout_info, None, None)
                .result()?
        };

        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE_LAYOUT, pipeline_layout.0)?;

        Ok(PipelineLayout {
            context: self.context.clone(),
            raw: pipeline_layout,
        })
    }

    /// Creates a new graphics pipeline.
    pub fn create_graphics_pipeline(
        &mut self,
        name: &str,
        pipeline_info: vk::GraphicsPipelineCreateInfoBuilder,
    ) -> Result<GraphicsPipeline> {
        let pipeline = unsafe {
            self.context
                .device
                .create_graphics_pipelines(None, &[pipeline_info], None)
                .result()?[0]
        };

        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE, pipeline.0)?;

        Ok(GraphicsPipeline {
            context: self.context.clone(),
            raw: pipeline,
        })
    }

    /// Creates a new compute pipeline.
    pub fn create_compute_pipeline(
        &mut self,
        name: &str,
        pipeline_info: vk::ComputePipelineCreateInfoBuilder,
    ) -> Result<ComputePipeline> {
        let pipeline = unsafe {
            self.context
                .device
                .create_compute_pipelines(None, &[pipeline_info], None)
                .result()?[0]
        };

        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE, pipeline.0)?;

        Ok(ComputePipeline {
            context: self.context.clone(),
            raw: pipeline,
        })
    }

    /// Creates a descriptor pool.
    pub fn create_descriptor_pool(
        &mut self,
        descriptor: &DescriptorPoolDescriptor,
    ) -> Result<DescriptorPool> {
        let pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .max_sets(descriptor.max_sets)
            .pool_sizes(descriptor.pool_sizes);

        let pool_info = if let Some(flags) = descriptor.flags {
            pool_info.flags(flags)
        } else {
            pool_info
        };

        let pool = unsafe {
            self.context
                .device
                .create_descriptor_pool(&pool_info, None, None)
                .result()?
        };

        self.context
            .set_object_name(descriptor.name, vk::ObjectType::DESCRIPTOR_POOL, pool.0)?;

        Ok(DescriptorPool::new(self.context.clone(), pool))
    }

    /// Creates a descriptor set layout.
    pub fn create_descriptor_set_layout(
        &mut self,
        name: &str,
        layout_info: vk::DescriptorSetLayoutCreateInfoBuilder,
    ) -> Result<DescriptorSetLayout> {
        let layout = unsafe {
            self.context
                .device
                .create_descriptor_set_layout(&layout_info, None, None)
                .result()?
        };

        self.context
            .set_object_name(name, vk::ObjectType::DESCRIPTOR_SET_LAYOUT, layout.0)?;

        Ok(DescriptorSetLayout::new(self.context.clone(), layout))
    }

    /// Creates a new shader module using the provided SPIR-V code.
    pub fn create_shader_module(&self, name: &str, shader_data: &[u8]) -> Result<ShaderModule> {
        let code = erupt::utils::decode_spv(shader_data)?;
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&code);
        let module = unsafe {
            self.context
                .device
                .create_shader_module(&create_info, None, None)
                .result()?
        };

        self.context
            .set_object_name(name, vk::ObjectType::SHADER_MODULE, module.0)?;

        Ok(ShaderModule {
            context: self.context.clone(),
            raw: module,
        })
    }

    /// Creates a new buffer.
    pub fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<Buffer> {
        let mut families = Vec::with_capacity(3);

        if descriptor.queues.contains(vk::QueueFlags::COMPUTE) {
            families.push(self.compute_queue_family_index)
        }
        if descriptor.queues.contains(vk::QueueFlags::GRAPHICS) {
            families.push(self.graphic_queue_family_index)
        }
        if descriptor.queues.contains(vk::QueueFlags::TRANSFER) {
            families.push(self.transfer_queue_family_index)
        }

        // Removes dupes if queues share family ids.
        families.sort_unstable();
        families.dedup();

        let create_info = vk::BufferCreateInfoBuilder::new()
            .queue_family_indices(&families)
            .sharing_mode(descriptor.sharing_mode)
            .usage(descriptor.usage)
            .size(descriptor.size);

        let create_info = if let Some(flags) = descriptor.flags {
            create_info.flags(flags)
        } else {
            create_info
        };

        let raw = unsafe {
            self.context
                .device
                .create_buffer(&create_info, None, None)
                .result()?
        };

        #[cfg(debug_assertions)]
        self.context
            .set_object_name(&descriptor.name, vk::ObjectType::BUFFER, raw.0)?;

        let allocation = self.context.allocator.lock().allocate_memory_for_buffer(
            &self.context.device,
            raw,
            descriptor.memory_location,
        )?;

        let bind_infos = vk::BindBufferMemoryInfoBuilder::new()
            .buffer(raw)
            .memory(allocation.device_memory)
            .memory_offset(allocation.offset);

        unsafe {
            self.context
                .device
                .bind_buffer_memory2(&[bind_infos])
                .result()?
        };

        Ok(Buffer {
            context: self.context.clone(),
            raw,
            allocation,
        })
    }

    /// Creates a new image.
    pub fn create_image(&self, descriptor: &ImageDescriptor) -> Result<Image> {
        let mut families = Vec::with_capacity(3);

        if descriptor.queues.contains(vk::QueueFlags::COMPUTE) {
            families.push(self.compute_queue_family_index)
        }
        if descriptor.queues.contains(vk::QueueFlags::GRAPHICS) {
            families.push(self.graphic_queue_family_index)
        }
        if descriptor.queues.contains(vk::QueueFlags::TRANSFER) {
            families.push(self.transfer_queue_family_index)
        }

        // Removes dupes if queues share family ids.
        families.sort_unstable();
        families.dedup();

        let create_info = vk::ImageCreateInfoBuilder::new()
            .queue_family_indices(&families)
            .sharing_mode(descriptor.sharing_mode)
            .usage(descriptor.usage)
            .extent(descriptor.extent)
            .mip_levels(descriptor.mip_levels)
            .format(descriptor.format)
            .image_type(descriptor.image_type)
            .array_layers(descriptor.array_layers)
            .samples(descriptor.samples)
            .tiling(descriptor.tiling)
            .initial_layout(descriptor.initial_layout);

        let create_info = if let Some(flags) = descriptor.flags {
            create_info.flags(flags)
        } else {
            create_info
        };

        let raw = unsafe {
            self.context
                .device
                .create_image(&create_info, None, None)
                .result()?
        };

        #[cfg(debug_assertions)]
        self.context
            .set_object_name(&descriptor.name, vk::ObjectType::IMAGE, raw.0)?;

        let allocation = self.context.allocator.lock().allocate_memory_for_image(
            &self.context.device,
            raw,
            descriptor.memory_location,
        )?;

        let bind_infos = vk::BindImageMemoryInfoBuilder::new()
            .image(raw)
            .memory(allocation.device_memory)
            .memory_offset(allocation.offset);

        unsafe {
            self.context
                .device
                .bind_image_memory2(&[bind_infos])
                .result()?
        };

        Ok(Image {
            context: self.context.clone(),
            raw,
            allocation,
        })
    }

    /// Creates a new image.
    pub fn create_image_view(&self, descriptor: &ImageViewDescriptor) -> Result<ImageView> {
        let create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(descriptor.image.raw)
            .view_type(descriptor.view_type)
            .format(descriptor.format)
            .components(descriptor.components)
            .subresource_range(descriptor.subresource_range);

        let create_info = if let Some(flags) = descriptor.flags {
            create_info.flags(flags)
        } else {
            create_info
        };

        let raw = unsafe {
            self.context
                .device
                .create_image_view(&create_info, None, None)
                .result()?
        };

        #[cfg(debug_assertions)]
        self.context
            .set_object_name(&descriptor.name, vk::ObjectType::IMAGE_VIEW, raw.0)?;

        Ok(ImageView {
            context: self.context.clone(),
            raw,
        })
    }

    /// Creates a sampler.
    pub fn create_sampler(&self, descriptor: &SamplerDescriptor) -> Result<Sampler> {
        let create_info = vk::SamplerCreateInfoBuilder::new()
            .mag_filter(descriptor.mag_filter)
            .min_filter(descriptor.min_filter)
            .mipmap_mode(descriptor.mipmap_mode)
            .address_mode_u(descriptor.address_mode_u)
            .address_mode_v(descriptor.address_mode_v)
            .address_mode_w(descriptor.address_mode_w)
            .mipmap_mode(descriptor.mip_lod_bias)
            .anisotropy_enable(descriptor.anisotropy_enable)
            .max_anisotropy(descriptor.max_anisotropy)
            .min_lod(descriptor.min_lod)
            .max_lod(descriptor.min_lod)
            .unnormalized_coordinates(descriptor.unnormalized_coordinates);

        let create_info = if let Some(flags) = descriptor.flags {
            create_info.flags(flags)
        } else {
            create_info
        };

        let create_info = if let Some(op) = descriptor.compare_op {
            create_info.compare_enable(true).compare_op(op)
        } else {
            create_info.compare_enable(false)
        };

        let create_info = if let Some(color) = descriptor.border_color {
            create_info.border_color(color)
        } else {
            create_info
        };

        let raw = unsafe {
            self.context
                .device
                .create_sampler(&create_info, None, None)
                .result()?
        };

        #[cfg(debug_assertions)]
        self.context
            .set_object_name(&descriptor.name, vk::ObjectType::SAMPLER, raw.0)?;

        Ok(Sampler {
            context: self.context.clone(),
            raw,
        })
    }

    /// Creates a new timeline semaphore.
    pub fn create_timeline_semaphore(
        &self,
        name: &str,
        initial_value: u64,
    ) -> Result<TimelineSemaphore> {
        let mut create_info = vk::SemaphoreTypeCreateInfoBuilder::new()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let semaphore_info = vk::SemaphoreCreateInfoBuilder::new().extend_from(&mut create_info);
        let raw = unsafe {
            self.context
                .device
                .create_semaphore(&semaphore_info, None, None)
                .result()?
        };

        self.context
            .set_object_name(name, vk::ObjectType::SEMAPHORE, raw.0)?;

        Ok(TimelineSemaphore::new(self.context.clone(), raw))
    }

    /// Flush mapped memory. Used for CPU->GPU transfers.
    pub fn flush_mapped_memory(&self, allocation: &vk_alloc::Allocation) -> Result<()> {
        let ranges = [vk::MappedMemoryRangeBuilder::new()
            .memory(allocation.device_memory)
            .size(allocation.size)
            .offset(allocation.offset)];
        unsafe {
            self.context
                .device
                .flush_mapped_memory_ranges(&ranges)
                .result()?;
        };

        Ok(())
    }

    /// Invalidate mapped memory. Used for GPU->CPU transfers.
    pub fn invalidate_mapped_memory(&self, allocation: &vk_alloc::Allocation) -> Result<()> {
        let ranges = [vk::MappedMemoryRangeBuilder::new()
            .memory(allocation.device_memory)
            .size(allocation.size)
            .offset(allocation.offset)];
        unsafe {
            self.context
                .device
                .invalidate_mapped_memory_ranges(&ranges)
                .result()?;
        };

        Ok(())
    }
}

fn query_support_resizable_bar(
    instance: &Instance,
    device: vk::PhysicalDevice,
    device_properties: &vk::PhysicalDeviceProperties,
) -> BARSupport {
    if device_properties.device_type != vk::PhysicalDeviceType::INTEGRATED_GPU {
        let memory_properties = vk::PhysicalDeviceMemoryProperties2Builder::new().build();
        let memory_properties = unsafe {
            instance
                .raw
                .get_physical_device_memory_properties2(device, Some(memory_properties))
        };

        let heap_indices: Vec<u32> = memory_properties
            .memory_properties
            .memory_types
            .iter()
            .filter(|t| {
                t.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )
            })
            .map(|t| t.heap_index)
            .collect();

        for index in heap_indices.iter() {
            let property = memory_properties.memory_properties.memory_heaps[*index as usize];
            // Normally BAR is at most 256 MiB, everything more must be resizable BAR.
            if property.size > 268435456 {
                return BARSupport::ResizableBAR;
            }
        }

        if !heap_indices.is_empty() {
            return BARSupport::BAR;
        }
    }

    BARSupport::NotSupported
}
