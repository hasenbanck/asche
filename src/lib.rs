#![warn(missing_docs)]
//! Provides an abstraction layer above ash to easier use vulkan in rust with minimal dependencies.

use std::sync::Arc;

use ash::version::InstanceV1_0;
use ash::version::{DeviceV1_0, EntryV1_0};
use ash::vk::Queue;
use ash::{extensions::ext, vk};
#[cfg(feature = "tracing")]
use tracing::{info, level_filters::LevelFilter};

pub use error::AscheError;

/// Debug code for vulkan.
#[cfg(feature = "tracing")]
mod debug;
/// Crate errors.
mod error;

type Result<T> = std::result::Result<T, AscheError>;

/// Construct a `*const std::os::raw::c_char` from a string
#[macro_export]
macro_rules! cstr {
    ($s:expr) => {
        concat!($s, "\0") as *const str as *const ::std::os::raw::c_char
    };
}

/// Vulkan version.
pub enum VulkanVersion {
    /// Vulkan version 1.0
    V1,
    /// Vulkan version 1.1
    V1_1,
    /// Vulkan version 1.2
    V1_2,
}

impl From<VulkanVersion> for u32 {
    fn from(version: VulkanVersion) -> Self {
        match version {
            VulkanVersion::V1 => vk::make_version(1, 0, 0),
            VulkanVersion::V1_1 => vk::make_version(1, 1, 0),
            VulkanVersion::V1_2 => vk::make_version(1, 2, 0),
        }
    }
}

/// Describes how the adapter should be configured.
pub struct AdapterDescriptor {
    /// Name of the application.
    pub app_name: String,
    /// Version of the application. Use `ash::vk::make_version()` to create the version number.
    pub app_version: u32,
    /// Version of the Vulkan API to request.
    pub vulkan_version: VulkanVersion,
}

impl Default for AdapterDescriptor {
    fn default() -> Self {
        Self {
            app_name: "Undefined".to_string(),
            app_version: vk::make_version(0, 0, 0),
            vulkan_version: VulkanVersion::V1,
        }
    }
}

/// Handles the creation of graphic and compute devices. Can be dropped once a `Device` is created.
pub struct Adapter {
    instance: Arc<Instance>,
}

impl Adapter {
    /// Creates a new `Adapter`.
    pub fn new(descriptor: &AdapterDescriptor) -> Result<Self> {
        let instance = Arc::new(Instance::new(descriptor)?);
        Ok(Self { instance })
    }

    /// Creates a new `Device` from this `Adapter`.
    pub fn request_device(&self, device_type: vk::PhysicalDeviceType) -> Result<Device> {
        #[cfg(feature = "tracing")]
        {
            // Create the physical device
            let phys_devs = unsafe { self.instance.internal.enumerate_physical_devices()? };

            let mut chosen = None;
            self.find_physical_device(device_type, phys_devs, &mut chosen);

            if let Some((physical_device, physical_device_properties)) = chosen {
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
                    self.create_logical_device(physical_device)?;

                info!("Created logical device and queues");

                Ok(Device {
                    _instance: self.instance.clone(),
                    logical_device,
                    _graphics_queue: graphics_queue,
                    _transfer_queue: transfer_queue,
                    _compute_queue: compute_queue,
                })
            } else {
                Err(AscheError::RequestDeviceError)
            }
        }

        #[cfg(not(feature = "tracing"))]
        {
            // Create the physical device
            let phys_devs = unsafe { self.instance.internal.enumerate_physical_devices()? };

            let mut chosen = None;
            self.find_physical_device(device_type, phys_devs, &mut chosen);

            if let Some((physical_device, _physical_device_properties)) = chosen {
                let (logical_device, (graphics_queue, transfer_queue, compute_queue)) =
                    self.create_logical_device(physical_device)?;

                Ok(Device {
                    _instance: self.instance.clone(),
                    logical_device,
                    _graphics_queue: graphics_queue,
                    _transfer_queue: transfer_queue,
                    _compute_queue: compute_queue,
                })
            } else {
                Err(AscheError::RequestDeviceError)
            }
        }
    }

    fn find_physical_device(
        &self,
        device_type: vk::PhysicalDeviceType,
        phys_devs: Vec<vk::PhysicalDevice>,
        chosen: &mut Option<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)>,
    ) {
        for device in phys_devs {
            let properties = unsafe {
                self.instance
                    .internal
                    .get_physical_device_properties(device)
            };

            if properties.device_type == device_type {
                *chosen = Some((device, properties))
            }
        }
    }

    fn create_logical_device(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<(ash::Device, (Queue, Queue, Queue))> {
        let queue_family_properties = unsafe {
            self.instance
                .internal
                .get_physical_device_queue_family_properties(physical_device)
        };

        let graphics_queue_family_id =
            self.find_queue_family(vk::QueueFlags::GRAPHICS, &queue_family_properties)?;
        let transfer_queue_family_id =
            self.find_queue_family(vk::QueueFlags::TRANSFER, &queue_family_properties)?;
        let compute_queue_family_id =
            self.find_queue_family(vk::QueueFlags::COMPUTE, &queue_family_properties)?;

        // If some queue families point to the same ID, we need to create only one
        // `vk::DeviceQueueCreateInfo` for them. The following if cases match all cases.
        if graphics_queue_family_id == transfer_queue_family_id
            && transfer_queue_family_id != compute_queue_family_id
        {
            // Case: G=T,C
            let queue_infos = [
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_family_id)
                    .queue_priorities(&[1.0f32, 1.0f32])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(compute_queue_family_id)
                    .queue_priorities(&[0.0f32])
                    .build(),
            ];
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_layer_names(&self.instance.layers);
            let logical_device = unsafe {
                self.instance
                    .internal
                    .create_device(physical_device, &device_create_info, None)?
            };
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 1) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 0) };

            Ok((logical_device, (g_q, t_q, c_q)))
        } else if graphics_queue_family_id != transfer_queue_family_id
            && transfer_queue_family_id == compute_queue_family_id
        {
            // Case: G,T=C
            let queue_infos = [
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_family_id)
                    .queue_priorities(&[1.0f32])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_family_id)
                    .queue_priorities(&[1.0f32, 0.0f32])
                    .build(),
            ];
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_layer_names(&self.instance.layers);
            let logical_device = unsafe {
                self.instance
                    .internal
                    .create_device(physical_device, &device_create_info, None)?
            };
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 1) };

            Ok((logical_device, (g_q, t_q, c_q)))
        } else if graphics_queue_family_id == compute_queue_family_id
            && graphics_queue_family_id != transfer_queue_family_id
        {
            // Case: G=C,T
            let queue_infos = [
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_family_id)
                    .queue_priorities(&[1.0f32, 0.0f32])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_family_id)
                    .queue_priorities(&[1.0f32])
                    .build(),
            ];
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_layer_names(&self.instance.layers);
            let logical_device = unsafe {
                self.instance
                    .internal
                    .create_device(physical_device, &device_create_info, None)?
            };
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 1) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };

            Ok((logical_device, (g_q, t_q, c_q)))
        } else if graphics_queue_family_id == transfer_queue_family_id
            && transfer_queue_family_id == compute_queue_family_id
        {
            // Case: G=T=C
            let queue_infos = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_family_id)
                .queue_priorities(&[1.0f32, 1.0f32, 0.0f32])
                .build()];
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_layer_names(&self.instance.layers);
            let logical_device = unsafe {
                self.instance
                    .internal
                    .create_device(physical_device, &device_create_info, None)?
            };
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 1) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 2) };

            Ok((logical_device, (g_q, t_q, c_q)))
        } else {
            // Case: G,T,C
            let queue_infos = [
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_family_id)
                    .queue_priorities(&[1.0f32])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_family_id)
                    .queue_priorities(&[1.0f32])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(compute_queue_family_id)
                    .queue_priorities(&[0.0f32])
                    .build(),
            ];
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_layer_names(&self.instance.layers);
            let logical_device = unsafe {
                self.instance
                    .internal
                    .create_device(physical_device, &device_create_info, None)?
            };
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 0) };

            Ok((logical_device, (g_q, t_q, c_q)))
        }
    }

    fn find_queue_family(
        &self,
        target_family: vk::QueueFlags,
        queue_family_properties: &[vk::QueueFamilyProperties],
    ) -> Result<u32> {
        let mut queue_id = None;
        for (id, family) in queue_family_properties.iter().enumerate() {
            match target_family {
                vk::QueueFlags::GRAPHICS => {
                    if family.queue_count > 0
                        && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
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
                vk::QueueFlags::GRAPHICS => Err(AscheError::QueueFamilyNotFound("graphic")),
                vk::QueueFlags::TRANSFER => Err(AscheError::QueueFamilyNotFound("transfer")),
                vk::QueueFlags::COMPUTE => Err(AscheError::QueueFamilyNotFound("compute")),
                _ => panic!("Unhandled vk::QueueFlags value"),
            }
        }
    }
}

/// Wraps the Vulkan instance.
struct Instance {
    _entry: ash::Entry,
    pub(crate) internal: ash::Instance,
    layers: Vec<*const i8>,

    #[cfg(feature = "tracing")]
    debug_util: ext::DebugUtils,
    #[cfg(feature = "tracing")]
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl Instance {
    /// Creates a new `Instance`.
    pub fn new(descriptor: &AdapterDescriptor) -> Result<Self> {
        let entry = ash::Entry::new()?;

        let engine_name = std::ffi::CString::new("asche")?;
        let app_name = std::ffi::CString::new(descriptor.app_name.clone())?;

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(descriptor.app_version)
            .engine_name(&engine_name)
            .engine_version(vk::make_version(0, 1, 0))
            .api_version(vk::make_version(1, 2, 0));

        // Activate all needed instance layers and extensions.
        let layers: Vec<*const i8> = vec![cstr!("VK_LAYER_KHRONOS_validation")];
        let extensions: Vec<*const i8> = vec![ext::DebugUtils::name().as_ptr()];

        #[cfg(feature = "tracing")]
        {
            let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(vulkan_log_level())
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                )
                .pfn_user_callback(Some(debug::debug_utils_callback));

            let create_info = vk::InstanceCreateInfo::builder()
                .push_next(&mut debug_create_info)
                .application_info(&app_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions);

            let instance = unsafe { entry.create_instance(&create_info, None)? };

            let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
            let utils_messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&debug_create_info, None)? };

            Ok(Self {
                _entry: entry,
                internal: instance,
                layers,
                debug_util: debug_utils,
                debug_messenger: utils_messenger,
            })
        }
        #[cfg(not(feature = "tracing"))]
        {
            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions);

            let instance = unsafe { entry.create_instance(&create_info, None)? };

            Ok(Self {
                _entry: entry,
                internal: instance,
                layers,
            })
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            #[cfg(feature = "tracing")]
            self.debug_util
                .destroy_debug_utils_messenger(self.debug_messenger, None);

            self.internal.destroy_instance(None)
        };
    }
}

/// Abstracts a Vulkan device.
pub struct Device {
    _instance: Arc<Instance>,
    logical_device: ash::Device,
    _graphics_queue: vk::Queue,
    _transfer_queue: vk::Queue,
    _compute_queue: vk::Queue,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.logical_device.destroy_device(None) };
    }
}

#[cfg(feature = "tracing")]
fn vulkan_log_level() -> vk::DebugUtilsMessageSeverityFlagsEXT {
    match tracing::level_filters::STATIC_MAX_LEVEL {
        LevelFilter::OFF => vk::DebugUtilsMessageSeverityFlagsEXT::empty(),
        LevelFilter::ERROR => vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        LevelFilter::WARN => {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        }
        LevelFilter::INFO => {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
        }
        LevelFilter::DEBUG => {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        }
        LevelFilter::TRACE => vk::DebugUtilsMessageSeverityFlagsEXT::all(),
    }
}
