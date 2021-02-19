use std::rc::Rc;

use ash::version::DeviceV1_0;
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, info};

use crate::instance::Instance;
use crate::swapchain::{Swapchain, SwapchainDescriptor, SwapchainFrame};
use crate::{AscheError, Result};

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
    context: Rc<DeviceContext>,
    allocator: vk_alloc::Allocator,
    swapchain: Option<Swapchain>,
    graphics_queue: Queue,
    transfer_queue: Queue,
    compute_queue: Queue,
    swapchain_format: vk::Format,
    swapchain_color_space: vk::ColorSpaceKHR,
    presentation_mode: vk::PresentModeKHR,
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

        let context = DeviceContext {
            instance,
            logical_device,
            physical_device,
        };

        let mut device = Device {
            context: Rc::new(context),
            allocator,
            graphics_queue,
            transfer_queue,
            compute_queue,
            presentation_mode: descriptor.presentation_mode,
            swapchain_format: descriptor.swapchain_format,
            swapchain_color_space: descriptor.swapchain_color_space,
            swapchain: None,
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

    /// Queues the frame in the presentation queue.
    pub fn queue_frame(&self, frame: SwapchainFrame) -> Result<()> {
        let swapchain = &self
            .swapchain
            .as_ref()
            .ok_or(AscheError::SwapchainNotInitialized)?;
        swapchain.queue_frame(frame, &self.graphics_queue)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // TODO destroy custom VK resources.
    }
}

/// The context for a device.
pub struct DeviceContext {
    pub(crate) instance: Instance,
    pub(crate) logical_device: ash::Device,
    pub(crate) physical_device: vk::PhysicalDevice,
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe { self.logical_device.destroy_device(None) };
    }
}
