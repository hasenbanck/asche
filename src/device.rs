use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::Handle;
#[cfg(feature = "tracing")]
use tracing::{debug, info};

use crate::command::CommandPool;
use crate::context::Context;
use crate::instance::Instance;
use crate::swapchain::{Swapchain, SwapchainDescriptor, SwapchainFrame};
use crate::{
    AscheError, CommandBuffer, Pipeline, PipelineLayout, Queue, QueueType, RenderPass, Result,
    ShaderModule, WaitForQueueType,
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
    allocator: vk_alloc::Allocator,
    swapchain: Option<Swapchain>,
    graphics_queue: Queue,
    transfer_queue: Queue,
    compute_queue: Queue,
    swapchain_format: vk::Format,
    swapchain_color_space: vk::ColorSpaceKHR,
    presentation_mode: vk::PresentModeKHR,
    command_pool_counter: u64,
    compute_fence: vk::Fence,
    graphics_fence: vk::Fence,
    transfer_fence: vk::Fence,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_fence(self.compute_fence, None)
        };
        unsafe {
            self.context
                .logical_device
                .destroy_fence(self.graphics_fence, None)
        };
        unsafe {
            self.context
                .logical_device
                .destroy_fence(self.transfer_fence, None)
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

        let (logical_device, (graphics_queue, transfer_queue, compute_queue)) =
            instance.create_logical_device(physical_device, &descriptor.queue_priority)?;

        #[cfg(feature = "tracing")]
        info!("Created logical device and queues");

        let allocator = vk_alloc::Allocator::new(
            &instance.raw,
            physical_device,
            &logical_device,
            &vk_alloc::AllocatorDescriptor::default(),
        );

        #[cfg(feature = "tracing")]
        debug!("Created default memory allocator");

        let context = Arc::new(Context {
            instance,
            logical_device,
            physical_device,
        });

        let fence_info = vk::FenceCreateInfo::builder();
        let compute_fence = unsafe { context.logical_device.create_fence(&fence_info, None)? };
        let graphics_fence = unsafe { context.logical_device.create_fence(&fence_info, None)? };
        let transfer_fence = unsafe { context.logical_device.create_fence(&fence_info, None)? };

        let mut device = Device {
            context,
            allocator,
            graphics_queue,
            transfer_queue,
            compute_queue,
            presentation_mode: descriptor.presentation_mode,
            swapchain_format: descriptor.swapchain_format,
            swapchain_color_space: descriptor.swapchain_color_space,
            swapchain: None,
            command_pool_counter: 0,
            compute_fence,
            graphics_fence,
            transfer_fence,
        };

        device.recreate_swapchain(None)?;

        Ok(device)
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

    /// Executes command buffers on a queue.
    pub fn execute(&self, queue_type: QueueType, command_buffers: &[&CommandBuffer]) -> Result<()> {
        let command_buffers: Vec<vk::CommandBuffer> = command_buffers
            .iter()
            .map(|buffer| buffer.encoder.buffer)
            .collect();

        let info = vk::SubmitInfo::builder().command_buffers(&command_buffers);

        unsafe {
            match queue_type {
                QueueType::Compute => {
                    self.context
                        .logical_device
                        .reset_fences(&[self.compute_fence])?;
                    self.context.logical_device.queue_submit(
                        self.compute_queue.raw,
                        &[info.build()],
                        self.compute_fence,
                    )?;
                }
                QueueType::Graphics => {
                    self.context
                        .logical_device
                        .reset_fences(&[self.graphics_fence])?;
                    self.context.logical_device.queue_submit(
                        self.graphics_queue.raw,
                        &[info.build()],
                        self.graphics_fence,
                    )?;
                }
                QueueType::Transfer => {
                    self.context
                        .logical_device
                        .reset_fences(&[self.transfer_fence])?;
                    self.context.logical_device.queue_submit(
                        self.transfer_queue.raw,
                        &[info.build()],
                        self.transfer_fence,
                    )?;
                }
            }
        };

        Ok(())
    }

    /// Waits until the execution on a queue has finished.
    pub fn wait(&self, queue_type: WaitForQueueType) -> Result<()> {
        let fences: Vec<vk::Fence> = match queue_type {
            WaitForQueueType::All => {
                vec![self.compute_fence, self.graphics_fence, self.transfer_fence]
            }
            WaitForQueueType::Compute => vec![self.compute_fence],
            WaitForQueueType::Graphics => vec![self.graphics_fence],
            WaitForQueueType::Transfer => vec![self.transfer_fence],
        };

        unsafe {
            self.context
                .logical_device
                .wait_for_fences(&fences, true, u64::MAX)?
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

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

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
