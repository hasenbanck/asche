use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, info};

use crate::context::Context;
use crate::Result;

/// Abstracts a Vulkan queue.
pub struct Queue {
    pub(crate) family_index: u32,
    pub(crate) inner: vk::Queue,
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
    context: Arc<Context>,
    logical_device: ash::Device,
    _graphics_queue: Queue,
    _transfer_queue: Queue,
    _compute_queue: Queue,
    allocator: vk_alloc::Allocator,
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
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

            let (swapchain, swapchain_loader) = context.create_swapchain(
                physical_device,
                &logical_device,
                &graphics_queue,
                descriptor.swapchain_format,
                descriptor.swapchain_color_space,
                descriptor.presentation_mode,
            )?;

            info!(
                "Created swapchain with format {:?} and color space {:?}",
                descriptor.swapchain_format, descriptor.swapchain_color_space
            );

            let allocator = vk_alloc::Allocator::new(
                &context.instance,
                physical_device,
                &logical_device,
                &vk_alloc::AllocatorDescriptor::default(),
            );
            debug!("Created default memory allocator");

            Ok(Device {
                context,
                logical_device,
                _graphics_queue: graphics_queue,
                _transfer_queue: transfer_queue,
                _compute_queue: compute_queue,
                allocator,
                swapchain_loader,
                swapchain,
            })
        }

        #[cfg(not(feature = "tracing"))]
        {
            let (physical_device, _) = self.find_physical_device(descriptor.device_type)?;

            let (logical_device, (graphics_queue, transfer_queue, compute_queue)) =
                self.create_logical_device(physical_device, &descriptor.queue_priority)?;

            let (swapchain, swapchain_loader) = self.create_swapchain(
                physical_device,
                &logical_device,
                &graphics_queue,
                &descriptor,
            )?;

            let allocator = vk_alloc::Allocator::new(
                &context.instance,
                physical_device,
                &logical_device,
                &vk_alloc::AllocatorDescriptor::default(),
            );

            Ok(Device {
                context,
                logical_device,
                _graphics_queue: graphics_queue,
                _transfer_queue: transfer_queue,
                _compute_queue: compute_queue,
                allocator,
                swapchain_loader,
                swapchain,
            })
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.logical_device.destroy_device(None);
        };
    }
}
