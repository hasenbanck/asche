use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::Arc;

use erupt::{vk, ExtendableFromConst};
#[cfg(feature = "tracing")]
use tracing1::{error, info};

use crate::context::Context;
use crate::instance::Instance;
use crate::query::QueryPool;
use crate::semaphore::TimelineSemaphore;
use crate::{
    AccelerationStructure, AscheError, BinarySemaphore, Buffer, BufferDescriptor, BufferView,
    BufferViewDescriptor, ComputePipeline, ComputeQueue, DeferredOperation, DescriptorPool,
    DescriptorPoolDescriptor, DescriptorSetLayout, Fence, GraphicsPipeline, GraphicsQueue, Image,
    ImageDescriptor, ImageView, ImageViewDescriptor, PipelineLayout, RayTracingPipeline,
    RenderPass, Result, Sampler, SamplerDescriptor, ShaderModule, Swapchain, TransferQueue,
};

/// Defines the configuration of the queues. Each vector entry defines the priority of a queue.
#[derive(Clone, Debug)]
pub struct QueueConfiguration {
    /// Specifies the priorities and amount of compute queues.
    pub compute_queues: Vec<f32>,
    /// Specifies the priorities and amount of graphics queues.
    pub graphics_queues: Vec<f32>,
    /// Specifies the priorities and amount of transfer queues.
    pub transfer_queues: Vec<f32>,
}

/// Describes how the device should be configured.
#[derive(Clone, Debug)]
pub struct DeviceConfiguration<'a> {
    /// The device type that is requested.
    pub device_type: vk::PhysicalDeviceType,
    /// The image format of the swapchain.
    pub swapchain_format: vk::Format,
    /// The color space of the swapchain.
    pub swapchain_color_space: vk::ColorSpaceKHR,
    /// The presentation mode of the swap chain.
    pub presentation_mode: vk::PresentModeKHR,
    /// The configuration of the queues.
    pub queue_configuration: QueueConfiguration,
    /// Device extensions to load.
    pub extensions: Vec<*const c_char>,
    /// Vulkan 1.0 features.
    pub features_v1_0: Option<vk::PhysicalDeviceFeaturesBuilder<'a>>,
    /// Vulkan 1.1 features.
    pub features_v1_1: Option<vk::PhysicalDeviceVulkan11FeaturesBuilder<'a>>,
    /// Vulkan 1.2 features
    pub features_v1_2: Option<vk::PhysicalDeviceVulkan12FeaturesBuilder<'a>>,
    /// VK_EXT_robustness2 features
    pub features_robustness2: Option<vk::PhysicalDeviceRobustness2FeaturesEXTBuilder<'a>>,
    /// VK_KHR_ray_tracing_pipeline features
    pub features_raytracing: Option<vk::PhysicalDeviceRayTracingPipelineFeaturesKHRBuilder<'a>>,
    /// VK_KHR_acceleration_structure features
    pub features_acceleration_structure:
        Option<vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder<'a>>,
}

impl<'a> Default for DeviceConfiguration<'a> {
    fn default() -> Self {
        Self {
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
            swapchain_format: vk::Format::B8G8R8A8_SRGB,
            swapchain_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR,
            presentation_mode: vk::PresentModeKHR::FIFO_KHR,
            queue_configuration: QueueConfiguration {
                graphics_queues: vec![1.0],
                transfer_queues: vec![1.0],
                compute_queues: vec![1.0],
            },
            extensions: vec![],
            features_v1_0: None,
            features_v1_1: None,
            features_v1_2: None,
            features_robustness2: None,
            features_raytracing: None,
            features_acceleration_structure: None,
        }
    }
}

/// Shows if the device support access to the device memory using the base address register.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarSupport {
    /// Device doesn't support BAR.
    NotSupported,
    /// Device supports BAR.
    Bar,
    /// Device supports resizable BAR.
    ResizableBar,
}

impl std::fmt::Display for BarSupport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BarSupport::NotSupported => f.write_str("No BAR support"),
            BarSupport::Bar => f.write_str("BAR supported"),
            BarSupport::ResizableBar => f.write_str("Resizable BAR supported"),
        }
    }
}

/// Contains all queues that were created for the device.
#[derive(Debug)]
pub struct Queues {
    /// Contains the created compute queues.
    pub compute_queues: Vec<ComputeQueue>,
    /// Contains the created graphics queues.
    pub graphics_queues: Vec<GraphicsQueue>,
    /// Contains the created transfer queues.
    pub transfer_queues: Vec<TransferQueue>,
}

/// A Vulkan device.
///
/// Handles all resource creation. Command buffer and queue handling are handled by the `Queue`.
/// Swapchain and framebuffer handling are handled by the `Swapchain`.
#[derive(Debug)]
pub struct Device {
    compute_queue_family_index: u32,
    graphic_queue_family_index: u32,
    transfer_queue_family_index: u32,
    /// The type of the physical device.
    pub device_type: vk::PhysicalDeviceType,
    /// Shows if the device support access to the device memory using the base address register.
    pub resizable_bar_support: BarSupport,
    context: Arc<Context>,
}

impl Device {
    /// Creates a new device, a swapchain and multiple queues:
    ///  * Compute queues
    ///  * Graphics queues
    ///  * Transfer queues
    ///
    /// The compute and transfer queues use dedicated queue families if provided by the implementation.
    /// The graphics queues are guaranteed to be able to write the the surface.
    #[allow(unused_variables)]
    pub(crate) fn new(
        instance: Instance,
        configuration: DeviceConfiguration,
    ) -> Result<(Self, Swapchain, Queues)> {
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

        #[cfg(feature = "tracing")]
        info!("Creating logical device and queues");

        let (device, family_ids, queues) =
            instance.create_device(physical_device, configuration.clone())?;

        #[cfg(feature = "tracing")]
        info!("Creating Vulkan memory allocator");

        let allocator = vk_alloc::Allocator::new(
            &instance.raw,
            physical_device,
            &vk_alloc::AllocatorDescriptor::default(),
        )?;

        let compute_queue_family_index = family_ids[0];
        let graphic_queue_family_index = family_ids[1];
        let transfer_queue_family_index = family_ids[2];

        let context = Arc::new(Context::new(instance, device, physical_device, allocator));

        let swapchain = Swapchain::new(
            context.clone(),
            graphic_queue_family_index,
            configuration.presentation_mode,
            configuration.swapchain_format,
            configuration.swapchain_color_space,
        )?;

        #[allow(clippy::as_conversions)]
        let compute_queues = queues[0]
            .iter()
            .enumerate()
            .map(|(i, queue)| {
                let q = ComputeQueue::new(context.clone(), family_ids[0], *queue);

                context
                    .set_object_name(
                        &format!("Compute Queue ({})", i),
                        vk::ObjectType::QUEUE,
                        q.raw.0 as u64,
                    )
                    .expect("can't set compute queue name");

                q
            })
            .collect::<_>();

        #[allow(clippy::as_conversions)]
        let graphics_queues = queues[1]
            .iter()
            .enumerate()
            .map(|(i, queue)| {
                let q = GraphicsQueue::new(context.clone(), family_ids[1], *queue);

                context
                    .set_object_name(
                        &format!("Graphics Queue ({})", i),
                        vk::ObjectType::QUEUE,
                        q.raw.0 as u64,
                    )
                    .expect("can't set graphics queue name");

                q
            })
            .collect::<_>();

        #[allow(clippy::as_conversions)]
        let transfer_queues = queues[2]
            .iter()
            .enumerate()
            .map(|(i, queue)| {
                let q = TransferQueue::new(context.clone(), family_ids[2], *queue);

                context
                    .set_object_name(
                        &format!("Transfer Queue ({})", i),
                        vk::ObjectType::QUEUE,
                        q.raw.0 as u64,
                    )
                    .expect("can't set graphics queue name");

                q
            })
            .collect::<_>();

        let device = Device {
            device_type: physical_device_properties.device_type,
            resizable_bar_support,
            context,
            compute_queue_family_index,
            graphic_queue_family_index,
            transfer_queue_family_index,
        };

        Ok((
            device,
            swapchain,
            Queues {
                compute_queues,
                graphics_queues,
                transfer_queues,
            },
        ))
    }

    /// Creates a new render pass.
    pub fn create_render_pass(
        &self,
        name: &str,
        renderpass_info: vk::RenderPassCreateInfo2Builder,
    ) -> Result<RenderPass> {
        let renderpass = unsafe {
            self.context
                .device
                .create_render_pass2(&renderpass_info, None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create a render pass: {}", err);
            AscheError::VkResult(err)
        })?;

        self.context
            .set_object_name(name, vk::ObjectType::RENDER_PASS, renderpass.0)?;

        Ok(RenderPass {
            context: self.context.clone(),
            raw: renderpass,
        })
    }

    /// Creates a new pipeline layout.
    pub fn create_pipeline_layout(
        &self,
        name: &str,
        pipeline_layout_info: vk::PipelineLayoutCreateInfoBuilder,
    ) -> Result<PipelineLayout> {
        let pipeline_layout = unsafe {
            self.context
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create a pipeline layout: {}", err);
            AscheError::VkResult(err)
        })?;

        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE_LAYOUT, pipeline_layout.0)?;

        Ok(PipelineLayout {
            context: self.context.clone(),
            raw: pipeline_layout,
        })
    }

    /// Creates a new graphics pipeline.
    pub fn create_graphics_pipeline(
        &self,
        name: &str,
        pipeline_info: vk::GraphicsPipelineCreateInfoBuilder,
    ) -> Result<GraphicsPipeline> {
        let pipelines = unsafe {
            self.context
                .device
                .create_graphics_pipelines(None, &[pipeline_info], None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create a graphic pipeline: {}", err);
            AscheError::VkResult(err)
        })?;

        assert_eq!(pipelines.len(), 1);
        let pipeline = pipelines[0];
        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE, pipeline.0)?;

        Ok(GraphicsPipeline {
            context: self.context.clone(),
            raw: pipeline,
        })
    }

    /// Creates a new raytracing pipeline.
    pub fn create_raytracing_pipeline(
        &self,
        name: &str,
        deferred_operation: Option<vk::DeferredOperationKHR>,
        pipeline_info: vk::RayTracingPipelineCreateInfoKHRBuilder,
    ) -> Result<RayTracingPipeline> {
        let pipelines = unsafe {
            self.context.device.create_ray_tracing_pipelines_khr(
                deferred_operation,
                None,
                &[pipeline_info],
                None,
            )
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create a ray tracing pipeline: {}", err);
            AscheError::VkResult(err)
        })?;

        assert_eq!(pipelines.len(), 1);
        let pipeline = pipelines[0];
        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE, pipeline.0)?;

        Ok(RayTracingPipeline {
            context: self.context.clone(),
            raw: pipeline,
        })
    }

    /// Creates a new compute pipeline.
    pub fn create_compute_pipeline(
        &self,
        name: &str,
        pipeline_info: vk::ComputePipelineCreateInfoBuilder,
    ) -> Result<ComputePipeline> {
        let pipelines = unsafe {
            self.context
                .device
                .create_compute_pipelines(None, &[pipeline_info], None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create a compute pipeline: {}", err);
            AscheError::VkResult(err)
        })?;

        assert_eq!(pipelines.len(), 1);
        let pipeline = pipelines[0];
        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE, pipeline.0)?;

        Ok(ComputePipeline {
            context: self.context.clone(),
            raw: pipeline,
        })
    }

    /// Creates a descriptor pool.
    pub fn create_descriptor_pool(
        &self,
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

        let pool = unsafe { self.context.device.create_descriptor_pool(&pool_info, None) }
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a descriptor pool: {}", err);
                AscheError::VkResult(err)
            })?;

        self.context
            .set_object_name(descriptor.name, vk::ObjectType::DESCRIPTOR_POOL, pool.0)?;

        Ok(DescriptorPool::new(self.context.clone(), pool))
    }

    /// Creates a descriptor set layout.
    pub fn create_descriptor_set_layout(
        &self,
        name: &str,
        layout_info: vk::DescriptorSetLayoutCreateInfoBuilder,
    ) -> Result<DescriptorSetLayout> {
        let layout = unsafe {
            self.context
                .device
                .create_descriptor_set_layout(&layout_info, None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create a descriptor set layout: {}", err);
            AscheError::VkResult(err)
        })?;

        self.context
            .set_object_name(name, vk::ObjectType::DESCRIPTOR_SET_LAYOUT, layout.0)?;

        Ok(DescriptorSetLayout::new(self.context.clone(), layout))
    }

    /// Creates a new shader module using the provided SPIR-V code.
    pub fn create_shader_module(&self, name: &str, shader_data: &[u8]) -> Result<ShaderModule> {
        let code = erupt::utils::decode_spv(shader_data)?;
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&code);
        let module = unsafe { self.context.device.create_shader_module(&create_info, None) }
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a shader module: {}", err);
                AscheError::VkResult(err)
            })?;

        self.context
            .set_object_name(name, vk::ObjectType::SHADER_MODULE, module.0)?;

        Ok(ShaderModule {
            context: self.context.clone(),
            raw: module,
        })
    }

    /// Creates a new buffer.
    pub fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<Buffer> {
        if descriptor.size == 0 {
            return Err(AscheError::BufferZeroSize);
        }

        let mut families: Vec<u32> = Vec::with_capacity(3);

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

        let raw =
            unsafe { self.context.device.create_buffer(&create_info, None) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a buffer: {}", err);
                AscheError::VkResult(err)
            })?;

        #[cfg(debug_assertions)]
        self.context
            .set_object_name(&descriptor.name, vk::ObjectType::BUFFER, raw.0)?;

        let allocation = self.context.allocator.allocate_memory_for_buffer(
            &self.context.device,
            raw,
            descriptor.memory_location,
        )?;

        let bind_infos = vk::BindBufferMemoryInfoBuilder::new()
            .buffer(raw)
            .memory(allocation.device_memory)
            .memory_offset(allocation.offset);

        unsafe { self.context.device.bind_buffer_memory2(&[bind_infos]) }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to bind buffer memory: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(Buffer::new(raw, allocation, self.context.clone()))
    }

    /// Creates a new buffer view.
    pub fn create_buffer_view(&self, descriptor: &BufferViewDescriptor) -> Result<BufferView> {
        let create_info = vk::BufferViewCreateInfoBuilder::new()
            .buffer(descriptor.buffer.raw)
            .format(descriptor.format)
            .offset(descriptor.offset)
            .range(descriptor.range);

        let create_info = if let Some(flags) = descriptor.flags {
            create_info.flags(flags)
        } else {
            create_info
        };

        let raw = unsafe { self.context.device.create_buffer_view(&create_info, None) }.map_err(
            |err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create an buffer view: {}", err);
                AscheError::VkResult(err)
            },
        )?;

        #[cfg(debug_assertions)]
        self.context
            .set_object_name(&descriptor.name, vk::ObjectType::BUFFER_VIEW, raw.0)?;

        Ok(BufferView::new(raw, self.context.clone()))
    }

    /// Creates a new image.
    pub fn create_image(&self, descriptor: &ImageDescriptor) -> Result<Image> {
        let mut families: Vec<u32> = Vec::with_capacity(3);

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

        let raw =
            unsafe { self.context.device.create_image(&create_info, None) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create an image: {}", err);
                AscheError::VkResult(err)
            })?;

        #[cfg(debug_assertions)]
        self.context
            .set_object_name(&descriptor.name, vk::ObjectType::IMAGE, raw.0)?;

        let allocation = self.context.allocator.allocate_memory_for_image(
            &self.context.device,
            raw,
            descriptor.memory_location,
        )?;

        let bind_infos = vk::BindImageMemoryInfoBuilder::new()
            .image(raw)
            .memory(allocation.device_memory)
            .memory_offset(allocation.offset);

        unsafe { self.context.device.bind_image_memory2(&[bind_infos]) }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to bind image memory: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(Image::new(raw, allocation, self.context.clone()))
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

        let raw = unsafe { self.context.device.create_image_view(&create_info, None) }.map_err(
            |err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create an image view: {}", err);
                AscheError::VkResult(err)
            },
        )?;

        #[cfg(debug_assertions)]
        self.context
            .set_object_name(&descriptor.name, vk::ObjectType::IMAGE_VIEW, raw.0)?;

        Ok(ImageView::new(raw, self.context.clone()))
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

        let raw =
            unsafe { self.context.device.create_sampler(&create_info, None) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a sampler: {}", err);
                AscheError::VkResult(err)
            })?;

        #[cfg(debug_assertions)]
        self.context
            .set_object_name(&descriptor.name, vk::ObjectType::SAMPLER, raw.0)?;

        Ok(Sampler::new(raw, self.context.clone()))
    }

    /// Creates a fence.
    pub fn create_fence(&self, name: &str) -> Result<Fence> {
        let fence_info = vk::FenceCreateInfoBuilder::new();
        let raw =
            unsafe { self.context.device.create_fence(&fence_info, None) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a fence: {}", err);
                AscheError::VkResult(err)
            })?;

        self.context
            .set_object_name(name, vk::ObjectType::FENCE, raw.0)?;

        Ok(Fence::new(self.context.clone(), raw))
    }

    /// Creates a new binary semaphore.
    pub fn create_binary_semaphore(&self, name: &str) -> Result<BinarySemaphore> {
        let create_info =
            vk::SemaphoreTypeCreateInfoBuilder::new().semaphore_type(vk::SemaphoreType::BINARY);
        let semaphore_info = vk::SemaphoreCreateInfoBuilder::new().extend_from(&create_info);
        let raw = unsafe { self.context.device.create_semaphore(&semaphore_info, None) }.map_err(
            |err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a binary semaphore: {}", err);
                AscheError::VkResult(err)
            },
        )?;

        self.context
            .set_object_name(name, vk::ObjectType::SEMAPHORE, raw.0)?;

        Ok(BinarySemaphore::new(self.context.clone(), raw))
    }

    /// Creates a new timeline semaphore.
    pub fn create_timeline_semaphore(
        &self,
        name: &str,
        initial_value: u64,
    ) -> Result<TimelineSemaphore> {
        let create_info = vk::SemaphoreTypeCreateInfoBuilder::new()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let semaphore_info = vk::SemaphoreCreateInfoBuilder::new().extend_from(&create_info);
        let raw = unsafe { self.context.device.create_semaphore(&semaphore_info, None) }.map_err(
            |err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a timeline semaphore: {}", err);
                AscheError::VkResult(err)
            },
        )?;

        self.context
            .set_object_name(name, vk::ObjectType::SEMAPHORE, raw.0)?;

        Ok(TimelineSemaphore::new(self.context.clone(), raw))
    }

    /// Creates a new query pool.
    pub fn create_query_pool(
        &self,
        name: &str,
        query_pool_info: vk::QueryPoolCreateInfoBuilder,
    ) -> Result<QueryPool> {
        let info = query_pool_info.build();
        let query_pool =
            unsafe { self.context.device.create_query_pool(&info, None) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a query pool: {}", err);
                AscheError::VkResult(err)
            })?;

        self.context
            .set_object_name(name, vk::ObjectType::QUERY_POOL, query_pool.0)?;

        Ok(QueryPool::new(query_pool, self.context.clone()))
    }

    /// Flush mapped memory. Used for CPU->GPU transfers.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkFlushMappedMemoryRanges.html)"]
    pub fn flush_mapped_memory(&self, allocation: &vk_alloc::Allocation) -> Result<()> {
        let ranges = [vk::MappedMemoryRangeBuilder::new()
            .memory(allocation.device_memory)
            .size(allocation.size)
            .offset(allocation.offset)];
        unsafe { self.context.device.flush_mapped_memory_ranges(&ranges) }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to flush a mapped memory range: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(())
    }

    /// Invalidate mapped memory. Used for GPU->CPU transfers.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkInvalidateMappedMemoryRanges.html)"]
    pub fn invalidate_mapped_memory(&self, allocation: &vk_alloc::Allocation) -> Result<()> {
        let ranges = [vk::MappedMemoryRangeBuilder::new()
            .memory(allocation.device_memory)
            .size(allocation.size)
            .offset(allocation.offset)];
        unsafe { self.context.device.invalidate_mapped_memory_ranges(&ranges) }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to invalidate a mapped memory range: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(())
    }

    /// Query ray tracing capture replay pipeline shader group handles.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetRayTracingCaptureReplayShaderGroupHandlesKHR.html)"]
    pub fn ray_tracing_capture_replay_shader_group_handles(
        &self,
        pipeline: vk::Pipeline,
        first_group: u32,
        group_count: u32,
        data: &[u8],
    ) -> Result<()> {
        #[allow(clippy::as_conversions)]
        unsafe {
            self.context
                .device
                .get_ray_tracing_capture_replay_shader_group_handles_khr(
                    pipeline,
                    first_group,
                    group_count,
                    data.len(),
                    data.as_ptr() as *mut c_void,
                )
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!(
                "Unable to query ray tracing capture replay pipeline shader group handles: {}",
                err
            );
            AscheError::VkResult(err)
        })?;

        Ok(())
    }

    /// Query ray tracing pipeline shader group handles.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetRayTracingShaderGroupHandlesKHR.html)"]
    pub fn ray_tracing_shader_group_handles(
        &self,
        pipeline: vk::Pipeline,
        first_group: u32,
        group_count: u32,
        data: &[u8],
    ) -> Result<()> {
        #[allow(clippy::as_conversions)]
        unsafe {
            self.context
                .device
                .get_ray_tracing_shader_group_handles_khr(
                    pipeline,
                    first_group,
                    group_count,
                    data.len(),
                    data.as_ptr() as *mut c_void,
                )
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!(
                "Unable to query ray tracing pipeline shader group handles: {}",
                err
            );
            AscheError::VkResult(err)
        })?;

        Ok(())
    }

    /// Query ray tracing pipeline shader group shader stack size.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetRayTracingShaderGroupStackSizeKHR.html)"]
    pub fn ray_tracing_shader_group_stack_size(
        &self,
        pipeline: vk::Pipeline,
        group: u32,
        group_shader: vk::ShaderGroupShaderKHR,
    ) -> u64 {
        unsafe {
            self.context
                .device
                .get_ray_tracing_shader_group_stack_size_khr(pipeline, group, group_shader)
        }
    }

    /// Returns properties of a physical device.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetPhysicalDeviceProperties2.html)"]
    pub fn physical_device_properties(
        &self,
        properties: vk::PhysicalDeviceProperties2Builder,
    ) -> vk::PhysicalDeviceProperties2 {
        unsafe {
            self.context.instance.raw.get_physical_device_properties2(
                self.context.physical_device,
                Some(properties.build()),
            )
        }
    }

    /// Create a deferred operation handle.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateDeferredOperationKHR.html)"]
    pub fn create_deferred_operation(&self, name: &str) -> Result<DeferredOperation> {
        let operation = unsafe { self.context.device.create_deferred_operation_khr(None) }
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a deferred operation handle: {}", err);
                AscheError::VkResult(err)
            })?;

        self.context
            .set_object_name(name, vk::ObjectType::DEFERRED_OPERATION_KHR, operation.0)?;

        Ok(DeferredOperation::new(operation, self.context.clone()))
    }

    /// Build an acceleration structure on the host.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkBuildAccelerationStructuresKHR.html)"]
    pub fn build_acceleration_structures(
        &self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHRBuilder],
        build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ) -> Result<()> {
        #[allow(clippy::as_conversions)]
        let build_range_infos = build_range_infos
            .iter()
            .map(|r| r as *const vk::AccelerationStructureBuildRangeInfoKHR)
            .collect::<Vec<*const vk::AccelerationStructureBuildRangeInfoKHR>>();

        unsafe {
            self.context
                .device
                .build_acceleration_structures_khr(None, infos, &build_range_infos)
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

    /// Create a new acceleration structure object.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateAccelerationStructureKHR.html)"]
    pub fn create_acceleration_structure(
        &self,
        name: &str,
        create_info: &vk::AccelerationStructureCreateInfoKHR,
    ) -> Result<AccelerationStructure> {
        let structure = unsafe {
            self.context
                .device
                .create_acceleration_structure_khr(create_info, None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!(
                "Unable to create a new acceleration structure object: {}",
                err
            );
            AscheError::VkResult(err)
        })?;

        self.context.set_object_name(
            name,
            vk::ObjectType::ACCELERATION_STRUCTURE_KHR,
            structure.0,
        )?;

        Ok(AccelerationStructure::new(structure, self.context.clone()))
    }

    /// Retrieve the required size for an acceleration structure.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html)"]
    pub fn acceleration_structure_build_sizes(
        &self,
        build_type: vk::AccelerationStructureBuildTypeKHR,
        build_info: &vk::AccelerationStructureBuildGeometryInfoKHR,
        max_primitive_counts: &[u32],
    ) -> vk::AccelerationStructureBuildSizesInfoKHR {
        unsafe {
            self.context
                .device
                .get_acceleration_structure_build_sizes_khr(
                    build_type,
                    build_info,
                    max_primitive_counts,
                )
        }
    }

    ///  Check if a serialized acceleration structure is compatible with the current device.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetDeviceAccelerationStructureCompatibilityKHR.html)"]
    pub fn device_acceleration_structure_compatibility(
        &self,
        version_info: &vk::AccelerationStructureVersionInfoKHR,
    ) -> vk::AccelerationStructureCompatibilityKHR {
        unsafe {
            self.context
                .device
                .get_device_acceleration_structure_compatibility_khr(version_info)
        }
    }

    /// Copy an acceleration structure on the host.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCopyAccelerationStructureKHR.html)"]
    pub fn copy_acceleration_structure(
        &self,
        info: &vk::CopyAccelerationStructureInfoKHRBuilder,
    ) -> Result<()> {
        unsafe {
            self.context
                .device
                .copy_acceleration_structure_khr(None, info)
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
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCopyAccelerationStructureToMemoryKHR.html)"]
    pub fn copy_acceleration_structure_to_memory(
        &self,
        info: &vk::CopyAccelerationStructureToMemoryInfoKHR,
    ) -> Result<()> {
        unsafe {
            self.context
                .device
                .copy_acceleration_structure_to_memory_khr(None, info)
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
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCopyMemoryToAccelerationStructureKHR.html)"]
    pub fn copy_memory_to_acceleration_structure(
        &self,
        info: &vk::CopyMemoryToAccelerationStructureInfoKHR,
    ) -> Result<()> {
        unsafe {
            self.context
                .device
                .copy_memory_to_acceleration_structure_khr(None, info)
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

    /// Update the contents of a descriptor set object.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkUpdateDescriptorSets.html)"]
    pub fn update_descriptor_sets(
        &self,
        descriptor_writes: &[vk::WriteDescriptorSetBuilder],
        descriptor_copies: &[vk::CopyDescriptorSetBuilder],
    ) {
        unsafe {
            self.context
                .device
                .update_descriptor_sets(descriptor_writes, descriptor_copies)
        }
    }

    /// Wait for a device to become idle.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDeviceWaitIdle.html)"]
    pub fn wait_idle(&self) -> Result<()> {
        unsafe { self.context.device.device_wait_idle() }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to wait for the device to become idle: {}", err);
            AscheError::VkResult(err)
        })
    }
}

fn query_support_resizable_bar(
    instance: &Instance,
    device: vk::PhysicalDevice,
    device_properties: &vk::PhysicalDeviceProperties,
) -> BarSupport {
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

        #[allow(clippy::as_conversions)]
        for index in heap_indices.iter() {
            let property = memory_properties.memory_properties.memory_heaps[*index as usize];
            // Normally BAR is at most 256 MiB, everything more must be resizable BAR.
            if property.size > 268435456 {
                return BarSupport::ResizableBar;
            }
        }

        if !heap_indices.is_empty() {
            return BarSupport::Bar;
        }
    }

    BarSupport::NotSupported
}
