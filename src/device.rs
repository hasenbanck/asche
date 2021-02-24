use std::ffi::CStr;
use std::io::Cursor;
use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::Handle;
use parking_lot::Mutex;
#[cfg(feature = "tracing")]
use tracing::info;

use crate::context::Context;
use crate::instance::Instance;
use crate::swapchain::{Swapchain, SwapchainDescriptor, SwapchainFrame};
use crate::{
    AscheError, ComputeQueue, GraphicsPipeline, GraphicsQueue, PipelineLayout, RenderPass, Result,
    ShaderModule, TransferQueue,
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
    pub extensions: Vec<&'static CStr>,
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
            swapchain_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            presentation_mode: vk::PresentModeKHR::FIFO,
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

/// A Vulkan device.
///
/// Handles all resource creation, swapchain and framebuffer handling. Command buffer and queue handling are handled by the `Queue`.
pub struct Device {
    context: Arc<Context>,
    swapchain: Option<Swapchain>,
    graphic_queue_family_index: u32,
    swapchain_format: vk::Format,
    swapchain_color_space: vk::ColorSpaceKHR,
    presentation_mode: vk::PresentModeKHR,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.context.logical_device.device_wait_idle().unwrap();
        };
    }
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
        }

        let presentation_mode = configuration.presentation_mode;
        let swapchain_format = configuration.swapchain_format;
        let swapchain_color_space = configuration.swapchain_color_space;

        let (logical_device, family_ids, queues) =
            instance.create_logical_device(physical_device, configuration)?;

        #[cfg(feature = "tracing")]
        info!("Created logical device and queues");

        let allocator = vk_alloc::Allocator::new(
            &instance.raw,
            physical_device,
            &logical_device,
            &vk_alloc::AllocatorDescriptor::default(),
        );

        let context = Arc::new(Context {
            instance,
            logical_device,
            physical_device,
            allocator: Mutex::new(allocator),
        });

        let graphic_queue_family_index = family_ids[1];
        let compute_queue = ComputeQueue::new(context.clone(), family_ids[0], queues[0])?;
        let graphics_queue = GraphicsQueue::new(context.clone(), family_ids[1], queues[1])?;
        let transfer_queue = TransferQueue::new(context.clone(), family_ids[2], queues[2])?;

        let mut device = Device {
            context,
            graphic_queue_family_index,
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
        let formats = unsafe {
            self.context
                .instance
                .surface_loader
                .get_physical_device_surface_formats(
                    self.context.physical_device,
                    self.context.instance.surface,
                )?
        };

        let capabilities = unsafe {
            self.context
                .instance
                .surface_loader
                .get_physical_device_surface_capabilities(
                    self.context.physical_device,
                    self.context.instance.surface,
                )?
        };

        let presentation_mode = unsafe {
            self.context
                .instance
                .surface_loader
                .get_physical_device_surface_present_modes(
                    self.context.physical_device,
                    self.context.instance.surface,
                )?
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

    /// Creates a new graphics pipeline.
    pub fn create_graphics_pipeline(
        &mut self,
        name: &str,
        pipeline_info: vk::GraphicsPipelineCreateInfoBuilder,
    ) -> Result<GraphicsPipeline> {
        let pipeline = unsafe {
            self.context.logical_device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info.build()],
                None,
            )?[0]
        };

        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE, pipeline.as_raw())?;

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
    ) -> Result<GraphicsPipeline> {
        let pipeline = unsafe {
            self.context.logical_device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info.build()],
                None,
            )?[0]
        };

        self.context
            .set_object_name(name, vk::ObjectType::PIPELINE, pipeline.as_raw())?;

        Ok(GraphicsPipeline {
            context: self.context.clone(),
            raw: pipeline,
        })
    }

    /// Creates a new shader module using the provided SPIR-V code.
    pub fn create_shader_module(&self, name: &str, shader_data: &[u8]) -> Result<ShaderModule> {
        let mut reader = Cursor::new(shader_data);
        let code = ash::util::read_spv(&mut reader)?;
        let create_info = vk::ShaderModuleCreateInfo::builder().code(&code);
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
}
