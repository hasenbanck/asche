use std::sync::Arc;

use ash::version::{DeviceV1_0, DeviceV1_2};
use ash::vk;
use ash::vk::Handle;
#[cfg(feature = "tracing")]
use tracing::info;

use crate::command::CommandPool;
use crate::context::Context;
use crate::instance::Instance;
use crate::swapchain::{Swapchain, SwapchainDescriptor, SwapchainFrame};
use crate::{
    AscheError, CommandBuffer, Pipeline, PipelineLayout, Queue, QueueType, RenderPass, Result,
    ShaderModule,
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
pub struct DeviceDescriptor {
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
}

impl Default for DeviceDescriptor {
    fn default() -> Self {
        Self {
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
            swapchain_format: vk::Format::B8G8R8A8_SRGB,
            swapchain_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            presentation_mode: vk::PresentModeKHR::FIFO,
            queue_priority: QueuePriorityDescriptor {
                graphics: 1.0,
                transfer: 1.0,
                compute: 1.0,
            },
        }
    }
}

/// Abstracts a Vulkan device.
pub struct Device {
    context: Arc<Context>,
    swapchain: Option<Swapchain>,
    graphics_queue: Queue,
    transfer_queue: Queue,
    compute_queue: Queue,
    swapchain_format: vk::Format,
    swapchain_color_space: vk::ColorSpaceKHR,
    presentation_mode: vk::PresentModeKHR,
    command_pool_counter: u64,
    timeline: vk::Semaphore,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.context.logical_device.device_wait_idle().unwrap();
            self.context
                .logical_device
                .destroy_semaphore(self.timeline, None)
        };
    }
}

impl Device {
    /// Creates a new device.
    pub(crate) fn new(instance: Instance, descriptor: &DeviceDescriptor) -> Result<Self> {
        let (physical_device, physical_device_properties) =
            instance.find_physical_device(descriptor.device_type)?;

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
        }

        let (logical_device, family_ids, queues) =
            instance.create_logical_device(physical_device, &descriptor.queue_priority)?;

        #[cfg(feature = "tracing")]
        info!("Created logical device and queues");

        let context = Arc::new(Context {
            instance,
            logical_device,
            physical_device,
        });

        let compute_queue = Queue {
            context: context.clone(),
            family_index: family_ids[0],
            raw: queues[0],
        };

        let graphics_queue = Queue {
            context: context.clone(),
            family_index: family_ids[1],
            raw: queues[1],
        };

        let transfer_queue = Queue {
            context: context.clone(),
            family_index: family_ids[2],
            raw: queues[2],
        };

        let mut create_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_info = vk::SemaphoreCreateInfo::builder().push_next(&mut create_info);
        let timeline = unsafe {
            context
                .logical_device
                .create_semaphore(&semaphore_info, None)?
        };

        let mut device = Device {
            context,
            graphics_queue,
            transfer_queue,
            compute_queue,
            presentation_mode: descriptor.presentation_mode,
            swapchain_format: descriptor.swapchain_format,
            swapchain_color_space: descriptor.swapchain_color_space,
            swapchain: None,
            command_pool_counter: 0,
            timeline,
        };

        device.recreate_swapchain(None)?;

        Ok(device)
    }

    /// Returns a reference to the context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Recreates the swapchain. Needs to be called if the surface has changed.
    pub fn recreate_swapchain(&mut self, window_extend: Option<vk::Extent2D>) -> Result<()> {
        let formats = unsafe {
            self.context
                .instance
                .surface_loader
                .get_physical_device_surface_formats(
                    self.context.physical_device,
                    self.context.instance.surface,
                )
        }?;

        let capabilities = unsafe {
            self.context
                .instance
                .surface_loader
                .get_physical_device_surface_capabilities(
                    self.context.physical_device,
                    self.context.instance.surface,
                )
        }?;

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
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            capabilities.current_transform
        };

        let format = formats
            .iter()
            .find(|f| {
                f.format == self.swapchain_format && f.color_space == self.swapchain_color_space
            })
            .ok_or(AscheError::SwapchainFormatIncompatible)?;

        let old_swapchain = self.swapchain.take();

        let swapchain = Swapchain::new(
            self.context.clone(),
            SwapchainDescriptor {
                graphic_queue_family_index: self.graphics_queue.family_index,
                extent,
                pre_transform,
                format: format.format,
                color_space: format.color_space,
                presentation_mode: self.presentation_mode,
                image_count,
            },
            old_swapchain,
        )?;

        #[cfg(feature = "tracing")]
        info!(
            "Created swapchain with format {:?}, color space {:?} and image count {}",
            self.swapchain_format, self.swapchain_color_space, image_count
        );

        self.swapchain = Some(swapchain);

        Ok(())
    }

    /// Gets the frame buffer for the given render pass and frame.
    pub fn get_frame_buffer(
        &self,
        render_pass: &RenderPass,
        frame: &SwapchainFrame,
    ) -> Result<vk::Framebuffer> {
        let swapchain = self
            .swapchain
            .as_ref()
            .ok_or(AscheError::SwapchainNotInitialized)?;

        Ok(swapchain.get_frame_buffer(render_pass.raw, frame.index))
    }

    /// Gets the next frame the program can render into.
    pub fn get_next_frame(&self) -> Result<SwapchainFrame> {
        let swapchain = self
            .swapchain
            .as_ref()
            .ok_or(AscheError::SwapchainNotInitialized)?;
        swapchain.get_next_frame()
    }

    /// Queues the frame in the presentation queue.8
    pub fn queue_frame(&self, frame: SwapchainFrame) -> Result<()> {
        let swapchain = &self
            .swapchain
            .as_ref()
            .ok_or(AscheError::SwapchainNotInitialized)?;
        swapchain.queue_frame(frame, &self.graphics_queue)
    }

    /// Executes a command buffer on a queue. Command buffer needs to originate from the same queue family as the queue.
    pub fn execute(&self, queue_type: QueueType, command_buffer: &CommandBuffer) -> Result<()> {
        // TODO support multiple timelines
        let (queue, semaphores) = match queue_type {
            QueueType::Compute => (self.compute_queue.raw, [self.timeline]),
            QueueType::Graphics => (self.graphics_queue.raw, [self.timeline]),
            QueueType::Transfer => (self.transfer_queue.raw, [self.timeline]),
        };

        let stage_masks = [ash::vk::PipelineStageFlags::ALL_COMMANDS];
        let command_buffers = [command_buffer.encoder.buffer];
        let wait_values = [command_buffer.timeline_wait_value];
        let signal_values = [command_buffer.timeline_signal_value];

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);

        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&stage_masks)
            .command_buffers(&command_buffers)
            .wait_semaphores(&semaphores)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline_info);

        unsafe {
            self.context.logical_device.queue_submit(
                queue,
                &[submit_info.build()],
                vk::Fence::null(),
            )?
        };

        Ok(())
    }

    /// TODO select queue.
    /// Query a timeline value.
    pub fn query_timeline_value(&self) -> Result<u64> {
        let value = unsafe {
            self.context
                .logical_device
                .get_semaphore_counter_value(self.timeline)?
        };
        Ok(value)
    }

    /// TODO select queue.
    /// Sets a timeline value.
    pub fn set_timeline_value(&self, timeline_value: u64) -> Result<()> {
        let signal_info = vk::SemaphoreSignalInfo::builder()
            .semaphore(self.timeline)
            .value(timeline_value);

        unsafe { self.context.logical_device.signal_semaphore(&signal_info)? };

        Ok(())
    }

    /// TODO select queue.
    /// Waits until the timeline has reached the value.
    pub fn wait_for_timeline_value(&self, timeline_value: u64) -> Result<()> {
        let semaphores = [self.timeline];
        let values = [timeline_value];
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(&semaphores)
            .values(&values);

        unsafe {
            self.context
                .logical_device
                .wait_semaphores(&wait_info, 5000000000)? // 5 sec timeout
        };

        Ok(())
    }

    /// Creates a new render pass.
    pub fn create_render_pass(
        &mut self,
        name: &str,
        renderpass_info: vk::RenderPassCreateInfoBuilder,
    ) -> Result<RenderPass> {
        let swapchain = self
            .swapchain
            .as_mut()
            .ok_or(AscheError::SwapchainNotInitialized)?;

        let renderpass = unsafe {
            self.context
                .logical_device
                .create_render_pass(&renderpass_info, None)?
        };

        swapchain.create_renderpass_framebuffers(renderpass)?;

        self.context
            .set_object_name(name, vk::ObjectType::RENDER_PASS, renderpass.as_raw())?;

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
                .logical_device
                .create_pipeline_layout(&pipeline_layout_info, None)?
        };

        self.context.set_object_name(
            name,
            vk::ObjectType::PIPELINE_LAYOUT,
            pipeline_layout.as_raw(),
        )?;

        Ok(PipelineLayout {
            context: self.context.clone(),
            raw: pipeline_layout,
        })
    }

    /// Creates a new pipeline.
    pub fn create_graphics_pipeline(
        &mut self,
        name: &str,
        pipeline_info: vk::GraphicsPipelineCreateInfoBuilder,
    ) -> Result<Pipeline> {
        let pipeline = unsafe {
            self.context.logical_device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info.build()],
                None,
            )?[0]
        };

        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE, pipeline.as_raw())?;

        Ok(Pipeline {
            context: self.context.clone(),
            raw: pipeline,
        })
    }

    /// Creates a new shader module using the provided SPIR-V code.
    pub fn create_shader_module(&self, name: &str, sprirv_code: &[u32]) -> Result<ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(sprirv_code);
        let module = unsafe {
            self.context
                .logical_device
                .create_shader_module(&create_info, None)?
        };

        self.context
            .set_object_name(name, vk::ObjectType::SHADER_MODULE, module.as_raw())?;

        Ok(ShaderModule {
            context: self.context.clone(),
            raw: module,
        })
    }

    /// Creates a new command pool. Pools are not cached and are owned by the caller.
    pub fn create_command_pool(&mut self, queue_type: QueueType) -> Result<CommandPool> {
        let queue = match queue_type {
            QueueType::Graphics => &self.graphics_queue,
            QueueType::Compute => &self.compute_queue,
            QueueType::Transfer => &self.transfer_queue,
        };

        let command_pool_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(queue.family_index);

        let command_pool = unsafe {
            self.context
                .logical_device
                .create_command_pool(&command_pool_info, None)?
        };

        let command_pool = CommandPool::new(
            self.context.clone(),
            command_pool,
            QueueType::Compute,
            self.command_pool_counter,
        )?;

        self.command_pool_counter += 1;

        Ok(command_pool)
    }
}
