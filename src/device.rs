use std::rc::Rc;

use ash::version::DeviceV1_0;
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, info};

use crate::instance::Instance;
use crate::swapchain::{Swapchain, SwapchainDescriptor, SwapchainFrame};
use crate::{
    AscheError, CommandPool, Context, Pipeline, PipelineLayout, RenderPass, Result, ShaderModule,
};

/// Abstracts a Vulkan queue.
pub struct Queue {
    pub(crate) family_index: u32,
    pub(crate) raw: vk::Queue,
}

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
    context: Rc<Context>,
    allocator: vk_alloc::Allocator,
    swapchain: Option<Swapchain>,
    graphics_queue: Queue,
    transfer_queue: Queue,
    compute_queue: Queue,
    swapchain_format: vk::Format,
    swapchain_color_space: vk::ColorSpaceKHR,
    presentation_mode: vk::PresentModeKHR,
    graphics_command_pools: Vec<Rc<CommandPool>>,
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

        let context = Rc::new(Context {
            instance,
            logical_device,
            physical_device,
        });

        let graphics_command_pools =
            Device::allocate_graphics_command_pools(&graphics_queue, &context)?;

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
            graphics_command_pools,
        };

        device.recreate_swapchain(None)?;

        Ok(device)
    }

    fn allocate_graphics_command_pools(
        graphics_queue: &Queue,
        context: &Rc<Context>,
    ) -> Result<Vec<Rc<CommandPool>>> {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_queue.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            context
                .logical_device
                .create_command_pool(&command_pool_info, None)?
        };

        let command_pool = Rc::new(CommandPool {
            context: context.clone(),
            raw: command_pool,
        });

        let graphic_command_pools = vec![command_pool];
        Ok(graphic_command_pools)
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
                extend: extent,
                pre_transform,
                format: format.format,
                color_space: format.color_space,
                presentation_mode: self.presentation_mode,
                image_count,
            },
            old_swapchain,
        )?;

        info!(
            "Created swapchain with format {:?}, color space {:?} and image count {}",
            self.swapchain_format, self.swapchain_color_space, image_count
        );

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

    /// Queues the frame in the presentation queue.8
    pub fn queue_frame(&self, frame: SwapchainFrame) -> Result<()> {
        let swapchain = &self
            .swapchain
            .as_ref()
            .ok_or(AscheError::SwapchainNotInitialized)?;
        swapchain.queue_frame(frame, &self.graphics_queue)
    }

    /// Creates a new render pass.
    pub fn create_render_pass(
        &mut self,
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

        swapchain.create_framebuffers(renderpass)?;

        Ok(RenderPass {
            context: self.context.clone(),
            raw: renderpass,
        })
    }

    /// Creates a new pipeline layout.
    pub fn create_pipeline_layout(
        &mut self,
        pipeline_layout_info: vk::PipelineLayoutCreateInfoBuilder,
    ) -> Result<PipelineLayout> {
        let pipeline_layout = unsafe {
            self.context
                .logical_device
                .create_pipeline_layout(&pipeline_layout_info, None)?
        };

        Ok(PipelineLayout {
            context: self.context.clone(),
            raw: pipeline_layout,
        })
    }

    /// Creates a new pipeline.
    pub fn create_graphics_pipeline(
        &mut self,
        pipeline_info: vk::GraphicsPipelineCreateInfoBuilder,
    ) -> Result<Pipeline> {
        let pipeline = unsafe {
            self.context.logical_device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info.build()],
                None,
            )?[0]
        };

        Ok(Pipeline {
            context: self.context.clone(),
            raw: pipeline,
        })
    }

    /// Creates a new shader module using the provided SPIR-V code.
    pub fn create_shader_module(&self, sprirv_code: &[u32]) -> Result<ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(sprirv_code);
        let module = unsafe {
            self.context
                .logical_device
                .create_shader_module(&create_info, None)?
        };

        Ok(ShaderModule {
            context: self.context.clone(),
            raw: module,
        })
    }

    // TODO create command pools. This should return an RCed command pool. We maybe need to RC the inner raw pool. We can then count how many pools are lent using the counter in the RC.
}
