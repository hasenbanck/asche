use crate::context::Context;
use crate::{AscheError, Result};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;
use std::ffi::CStr;

#[cfg(feature = "tracing")]
use tracing::{debug, error, info};

/// Abstracts a Vulkan queue.
pub struct Queue {
    family_index: u32,
    inner: vk::Queue,
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

/// Abstracts a Vulkan device. Currently only one device can be created.
pub struct Device {
    context: Context,
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
    pub fn new(context: Context, descriptor: &DeviceDescriptor) -> Result<Self> {
        #[cfg(feature = "tracing")]
        {
            let (physical_device, physical_device_properties) =
                Self::find_physical_device(&context, descriptor.device_type)?;

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
                Self::create_logical_device(&context, physical_device, &descriptor.queue_priority)?;

            info!("Created logical device and queues");

            let (swapchain, swapchain_loader) = Self::create_swapchain(
                &context,
                physical_device,
                &logical_device,
                &graphics_queue,
                &descriptor,
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
            let (physical_device, _) =
                Self::find_physical_device(&context, descriptor.device_type)?;

            let (logical_device, (graphics_queue, transfer_queue, compute_queue)) =
                Self::create_logical_device(&context, physical_device, &descriptor.queue_priority)?;

            let (swapchain, swapchain_loader) = Self::create_swapchain(
                &context,
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

    fn create_swapchain(
        context: &Context,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        graphic_queue: &Queue,
        descriptor: &DeviceDescriptor,
    ) -> Result<(vk::SwapchainKHR, ash::extensions::khr::Swapchain)> {
        let capabilities = unsafe {
            context
                .surface_loader
                .get_physical_device_surface_capabilities(physical_device, context.surface)
        }?;
        let formats = unsafe {
            context
                .surface_loader
                .get_physical_device_surface_formats(physical_device, context.surface)
        }?;

        #[cfg(feature = "tracing")]
        {
            let present_modes = unsafe {
                context
                    .surface_loader
                    .get_physical_device_surface_present_modes(physical_device, context.surface)
            }?;

            info!("Available surface capabilities:");
            info!("\tmin_image_count: {}", capabilities.min_image_count);
            info!("\tmax_image_count: {}", capabilities.max_image_count);
            info!(
                "\tmax_image_array_layers: {}",
                capabilities.max_image_array_layers
            );
            info!(
                "\tcurrent_extent: {}x{}",
                capabilities.current_extent.width, capabilities.current_extent.height
            );
            info!(
                "\tmin_image_extent: {}x{}",
                capabilities.min_image_extent.width, capabilities.min_image_extent.height
            );
            info!(
                "\tmax_image_extent: {}x{}",
                capabilities.max_image_extent.width, capabilities.max_image_extent.height
            );
            info!(
                "\tsupported_transforms: {:?}",
                capabilities.supported_transforms
            );
            info!("\tcurrent_transform: {:?}", capabilities.current_transform);
            info!(
                "\tsupported_composite_alpha: {:?}",
                capabilities.supported_composite_alpha
            );
            info!(
                "\tsupported_usage_flags: {:?}",
                capabilities.supported_usage_flags
            );

            info!("Available surface presentation modes: {:?}", present_modes);

            info!("Available surface formats:");
            formats.iter().for_each(|format| {
                info!("\t{:?} ({:?})", format.format, format.color_space);
            });
        }

        let format = formats
            .iter()
            .find(|format| {
                format.format == descriptor.swapchain_format
                    && format.color_space == descriptor.swapchain_color_space
            })
            .ok_or(AscheError::SwapchainFormatIncompatible)?;

        let family_index = &[graphic_queue.family_index];

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(context.surface)
            .min_image_count(
                3.max(capabilities.min_image_count)
                    .min(capabilities.max_image_count),
            )
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(family_index)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(descriptor.presentation_mode);
        let swapchain_loader =
            ash::extensions::khr::Swapchain::new(&context.instance, logical_device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };
        Ok((swapchain, swapchain_loader))
    }

    fn find_physical_device(
        context: &Context,
        device_type: vk::PhysicalDeviceType,
    ) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> {
        let physical_devices = unsafe { context.instance.enumerate_physical_devices()? };

        let mut chosen: Option<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> = None;
        for device in physical_devices {
            let properties = unsafe { context.instance.get_physical_device_properties(device) };

            if properties.device_type == device_type {
                chosen = Some((device, properties))
            }
        }
        let chosen = chosen.ok_or(AscheError::RequestDeviceError)?;

        Ok(chosen)
    }

    /// Creates a new logical device. Returns the logical device and three separate queues with the
    /// types vk::QueueFlags::GRAPHICS, vk::QueueFlags::TRANSFER and vk::QueueFlags::COMPUTE.
    fn create_logical_device(
        context: &Context,
        physical_device: vk::PhysicalDevice,
        priorities: &QueuePriorityDescriptor,
    ) -> Result<(ash::Device, (Queue, Queue, Queue))> {
        let queue_family_properties = unsafe {
            context
                .instance
                .get_physical_device_queue_family_properties(physical_device)
        };

        let graphics_queue_family_id = Self::find_queue_family(
            context,
            physical_device,
            vk::QueueFlags::GRAPHICS,
            &queue_family_properties,
        )?;
        let transfer_queue_family_id = Self::find_queue_family(
            context,
            physical_device,
            vk::QueueFlags::TRANSFER,
            &queue_family_properties,
        )?;
        let compute_queue_family_id = Self::find_queue_family(
            context,
            physical_device,
            vk::QueueFlags::COMPUTE,
            &queue_family_properties,
        )?;

        Self::create_queues_and_device(
            context,
            physical_device,
            priorities,
            graphics_queue_family_id,
            transfer_queue_family_id,
            compute_queue_family_id,
        )
    }

    fn create_queues_and_device(
        context: &Context,
        physical_device: vk::PhysicalDevice,
        priorities: &QueuePriorityDescriptor,
        graphics_queue_family_id: u32,
        transfer_queue_family_id: u32,
        compute_queue_family_id: u32,
    ) -> Result<(ash::Device, (Queue, Queue, Queue))> {
        // If some queue families point to the same ID, we need to create only one
        // `vk::DeviceQueueCreateInfo` for them.
        if graphics_queue_family_id == transfer_queue_family_id
            && transfer_queue_family_id != compute_queue_family_id
        {
            // Case: G=T,C
            let queue_infos = [
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_family_id)
                    .queue_priorities(&[priorities.graphics, priorities.transfer])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(compute_queue_family_id)
                    .queue_priorities(&[priorities.compute])
                    .build(),
            ];
            let logical_device = Self::create_device(context, physical_device, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 1) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 0) };

            Ok((
                logical_device,
                (
                    Queue {
                        family_index: graphics_queue_family_id,
                        inner: g_q,
                    },
                    Queue {
                        family_index: transfer_queue_family_id,
                        inner: t_q,
                    },
                    Queue {
                        family_index: compute_queue_family_id,
                        inner: c_q,
                    },
                ),
            ))
        } else if graphics_queue_family_id != transfer_queue_family_id
            && transfer_queue_family_id == compute_queue_family_id
        {
            // Case: G,T=C
            let queue_infos = [
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_family_id)
                    .queue_priorities(&[priorities.graphics])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_family_id)
                    .queue_priorities(&[priorities.transfer, priorities.compute])
                    .build(),
            ];
            let logical_device = Self::create_device(context, physical_device, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 1) };

            Ok((
                logical_device,
                (
                    Queue {
                        family_index: graphics_queue_family_id,
                        inner: g_q,
                    },
                    Queue {
                        family_index: transfer_queue_family_id,
                        inner: t_q,
                    },
                    Queue {
                        family_index: compute_queue_family_id,
                        inner: c_q,
                    },
                ),
            ))
        } else if graphics_queue_family_id == compute_queue_family_id
            && graphics_queue_family_id != transfer_queue_family_id
        {
            // Case: G=C,T
            let queue_infos = [
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_family_id)
                    .queue_priorities(&[priorities.graphics, priorities.compute])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_family_id)
                    .queue_priorities(&[priorities.transfer])
                    .build(),
            ];
            let logical_device = Self::create_device(context, physical_device, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 1) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };

            Ok((
                logical_device,
                (
                    Queue {
                        family_index: graphics_queue_family_id,
                        inner: g_q,
                    },
                    Queue {
                        family_index: transfer_queue_family_id,
                        inner: t_q,
                    },
                    Queue {
                        family_index: compute_queue_family_id,
                        inner: c_q,
                    },
                ),
            ))
        } else if graphics_queue_family_id == transfer_queue_family_id
            && transfer_queue_family_id == compute_queue_family_id
        {
            // Case: G=T=C
            let queue_infos = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_family_id)
                .queue_priorities(&[priorities.graphics, priorities.transfer, priorities.compute])
                .build()];
            let logical_device = Self::create_device(context, physical_device, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 1) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 2) };

            Ok((
                logical_device,
                (
                    Queue {
                        family_index: graphics_queue_family_id,
                        inner: g_q,
                    },
                    Queue {
                        family_index: transfer_queue_family_id,
                        inner: t_q,
                    },
                    Queue {
                        family_index: compute_queue_family_id,
                        inner: c_q,
                    },
                ),
            ))
        } else {
            // Case: G,T,C
            let queue_infos = [
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_family_id)
                    .queue_priorities(&[priorities.graphics])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_family_id)
                    .queue_priorities(&[priorities.transfer])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(compute_queue_family_id)
                    .queue_priorities(&[priorities.compute])
                    .build(),
            ];
            let logical_device = Self::create_device(context, physical_device, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 0) };

            Ok((
                logical_device,
                (
                    Queue {
                        family_index: graphics_queue_family_id,
                        inner: g_q,
                    },
                    Queue {
                        family_index: transfer_queue_family_id,
                        inner: t_q,
                    },
                    Queue {
                        family_index: compute_queue_family_id,
                        inner: c_q,
                    },
                ),
            ))
        }
    }

    fn create_device(
        context: &Context,
        physical_device: vk::PhysicalDevice,
        queue_infos: &[vk::DeviceQueueCreateInfo],
    ) -> Result<ash::Device> {
        let device_extensions = Self::create_device_extensions(context, physical_device)?;

        #[cfg(feature = "tracing")]
        {
            for extension in device_extensions.iter() {
                info!("Loading device extension: {:?}", extension);
            }
        }

        let features = vk::PhysicalDeviceFeatures::builder()
            .robust_buffer_access(true)
            .multi_draw_indirect(true)
            .full_draw_index_uint32(true)
            .tessellation_shader(true)
            .texture_compression_bc(true)
            .sampler_anisotropy(true)
            .shader_uniform_buffer_array_dynamic_indexing(true)
            .shader_sampled_image_array_dynamic_indexing(true)
            .shader_storage_buffer_array_dynamic_indexing(true)
            .shader_storage_image_array_dynamic_indexing(true);

        let mut features11 = vk::PhysicalDeviceVulkan11Features::builder();
        let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
            .draw_indirect_count(true)
            .descriptor_indexing(true)
            .imageless_framebuffer(true)
            .timeline_semaphore(true)
            .buffer_device_address(true);

        let layer_pointers = context
            .layers
            .iter()
            .map(|&s| s.as_ptr())
            .collect::<Vec<_>>();

        let extension_pointers = device_extensions
            .iter()
            .map(|&s| s.as_ptr())
            .collect::<Vec<_>>();

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_infos)
            .enabled_extension_names(&extension_pointers)
            .enabled_layer_names(&layer_pointers)
            .enabled_features(&features)
            .push_next(&mut features11)
            .push_next(&mut features12);

        let logical_device = unsafe {
            context
                .instance
                .create_device(physical_device, &device_create_info, None)?
        };

        Ok(logical_device)
    }

    fn create_device_extensions(
        context: &Context,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<&CStr>> {
        let mut extensions: Vec<&'static CStr> = Vec::new();

        extensions.push(ash::extensions::khr::Swapchain::name());

        // Only keep available extensions.
        let device_extensions = unsafe {
            context
                .instance
                .enumerate_device_extension_properties(physical_device)
        }?;

        extensions.retain(|&ext| {
            let found = device_extensions
                .iter()
                .any(|inst_ext| unsafe { CStr::from_ptr(inst_ext.extension_name.as_ptr()) == ext });
            if found {
                true
            } else {
                #[cfg(feature = "tracing")]
                error!("Unable to find device extension: {}", ext.to_string_lossy());
                false
            }
        });

        Ok(extensions)
    }

    fn find_queue_family(
        context: &Context,
        physical_device: vk::PhysicalDevice,
        target_family: vk::QueueFlags,
        queue_family_properties: &[vk::QueueFamilyProperties],
    ) -> Result<u32> {
        let mut queue_id = None;
        for (id, family) in queue_family_properties.iter().enumerate() {
            match target_family {
                vk::QueueFlags::GRAPHICS => {
                    if family.queue_count > 0
                        && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && unsafe {
                            context.surface_loader.get_physical_device_surface_support(
                                physical_device,
                                id as u32,
                                context.surface,
                            )?
                        }
                    {
                        queue_id = Some(id as u32);
                    }
                }
                vk::QueueFlags::TRANSFER => {
                    if family.queue_count > 0
                        && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && (queue_id.is_none()
                            || !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                && !family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    {
                        queue_id = Some(id as u32);
                    }
                }
                vk::QueueFlags::COMPUTE => {
                    if family.queue_count > 0
                        && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && (queue_id.is_none()
                            || !family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    {
                        queue_id = Some(id as u32);
                    }
                }
                _ => panic!("Unhandled vk::QueueFlags value"),
            }
        }

        if let Some(id) = queue_id {
            #[cfg(feature = "tracing")]
            info!("Found {:?} queue family with ID {}", target_family, id);

            Ok(id)
        } else {
            match target_family {
                vk::QueueFlags::GRAPHICS => {
                    Err(AscheError::QueueFamilyNotFound("graphic".to_string()))
                }
                vk::QueueFlags::TRANSFER => {
                    Err(AscheError::QueueFamilyNotFound("transfer".to_string()))
                }
                vk::QueueFlags::COMPUTE => {
                    Err(AscheError::QueueFamilyNotFound("compute".to_string()))
                }
                _ => panic!("Unhandled vk::QueueFlags value"),
            }
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
