use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, info};

use crate::context::Context;
use crate::swapchain::{Swapchain, SwapchainDescriptor};
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
    pub(crate) context: Arc<Context>,
    pub(crate) raw: ash::Device,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) graphics_queue: Queue,
    pub(crate) transfer_queue: Queue,
    pub(crate) compute_queue: Queue,
    pub(crate) allocator: vk_alloc::Allocator,
    pub(crate) swapchain_format: vk::Format,
    pub(crate) swapchain_color_space: vk::ColorSpaceKHR,
    pub(crate) presentation_mode: vk::PresentModeKHR,
    pub(crate) swapchain: Option<Swapchain>,
}

impl Device {
    /// Creates a new device.
    pub fn new(context: Arc<Context>, descriptor: &DeviceDescriptor) -> Result<Self> {
        #[cfg(feature = "tracing")]
        {
            let (physical_device, physical_device_properties) =
                context.find_physical_device(descriptor.device_type)?;

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

            let (logical_device, (graphics_queue, transfer_queue, compute_queue)) =
                context.create_logical_device(physical_device, &descriptor.queue_priority)?;

            info!("Created logical device and queues");

            let allocator = vk_alloc::Allocator::new(
                &context.instance,
                physical_device,
                &logical_device,
                &vk_alloc::AllocatorDescriptor::default(),
            );
            debug!("Created default memory allocator");

            let mut device = Device {
                context,
                raw: logical_device,
                physical: physical_device,
                graphics_queue,
                transfer_queue,
                compute_queue,
                allocator,
                swapchain_format: descriptor.swapchain_format,
                swapchain_color_space: descriptor.swapchain_color_space,
                presentation_mode: descriptor.presentation_mode,
                swapchain: None,
            };

            device.recreate_swapchain()?;

            Ok(device)
        }

        #[cfg(not(feature = "tracing"))]
        {
            let (physical_device, _) = self.find_physical_device(descriptor.device_type)?;

            let (logical_device, (graphics_queue, transfer_queue, compute_queue)) =
                self.create_logical_device(physical_device, &descriptor.queue_priority)?;

            let swapchain = context.create_swapchain(
                physical_device,
                &logical_device,
                &graphics_queue,
                descriptor.swapchain_format,
                descriptor.swapchain_color_space,
                descriptor.presentation_mode,
            )?;

            let allocator = vk_alloc::Allocator::new(
                &context.instance,
                physical_device,
                &logical_device,
                &vk_alloc::AllocatorDescriptor::default(),
            );

            Ok(Device {
                context,
                raw: logical_device,
                _graphics_queue: graphics_queue,
                _transfer_queue: transfer_queue,
                _compute_queue: compute_queue,
                allocator,
                swapchain,
            })
        }
    }

    // TODO get_current_frame()

    /// Recreates the swapchain. Needs to be called if the surface has changed.
    pub fn recreate_swapchain(&mut self) -> Result<()> {
        let formats = unsafe {
            self.context
                .surface_loader
                .get_physical_device_surface_formats(self.physical, self.context.surface)
        }?;

        let capabilities = unsafe {
            self.context
                .surface_loader
                .get_physical_device_surface_capabilities(self.physical, self.context.surface)
        }?;

        let image_count = 3
            .max(capabilities.min_image_count)
            .min(capabilities.max_image_count);

        let format = formats
            .iter()
            .find(|f| {
                f.format == self.swapchain_format && f.color_space == self.swapchain_color_space
            })
            .ok_or(AscheError::SwapchainFormatIncompatible)?;

        // TODO This works, but doesn't use the "old_swapchain" that Vulkan provides! Investigate how we would need to handle that.
        if let Some(mut old_swapchain) = self.swapchain.take() {
            old_swapchain.destroy(&self.raw)
        }

        let swapchain = Swapchain::new(
            &self,
            SwapchainDescriptor {
                graphic_queue_family_index: self.graphics_queue.family_index,
                extend: capabilities.current_extent,
                transform: capabilities.current_transform,
                format: format.format,
                color_space: format.color_space,
                presentation_mode: self.presentation_mode,
                image_count,
            },
            None,
        )?;

        info!(
            "Created swapchain with format {:?} and color space {:?}",
            self.swapchain_format, self.swapchain_color_space
        );

        self.swapchain = Some(swapchain);

        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            if let Some(swapchain) = &mut self.swapchain {
                swapchain.destroy(&self.raw);
            }
            self.raw.destroy_device(None);
        };
    }
}
