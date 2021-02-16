#![warn(missing_docs)]
//! Provides an abstraction layer above ash to easier use Vulkan in Rust with minimal dependencies.

use std::ffi::CStr;
use std::sync::Arc;

#[cfg(feature = "tracing")]
use ash::extensions::ext;
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
use raw_window_handle::RawWindowHandle;
#[cfg(feature = "tracing")]
use tracing::{debug, error, info, level_filters::LevelFilter, warn};

pub use error::AscheError;

/// Debug code for Vulkan.
#[cfg(feature = "tracing")]
mod debug;
/// Crate errors.
mod error;

type Result<T> = std::result::Result<T, AscheError>;

/// Construct a `*const std::os::raw::c_char` from a string
#[macro_export]
macro_rules! cstr {
    ($s:expr) => {
        concat!($s, "\0") as *const str as *const c_char
    };
}

/// Describes how the instance should be configured.
pub struct AdapterDescriptor {
    /// Name of the application.
    pub app_name: String,
    /// Version of the application. Use `ash::vk::make_version()` to create the version number.
    pub app_version: u32,
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

impl Default for AdapterDescriptor {
    fn default() -> Self {
        Self {
            app_name: "Undefined".to_string(),
            app_version: vk::make_version(0, 0, 0),
        }
    }
}

/// Describes how the device should be configured.
pub struct DeviceDescriptor {
    /// The device type that is requested.
    pub device_type: vk::PhysicalDeviceType,
    /// The priorities of the queues.
    pub queue_priority: QueuePriorityDescriptor,
}

impl Default for DeviceDescriptor {
    fn default() -> Self {
        Self {
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
            queue_priority: QueuePriorityDescriptor {
                graphics: 1.0,
                transfer: 1.0,
                compute: 1.0,
            },
        }
    }
}

/// Handles the creation of devices. Can be dropped once a `Device` is created.
pub struct Adapter(Arc<AshContext>);

impl Adapter {
    /// Creates a new `Adapter`.
    pub fn new(
        handle: &raw_window_handle::RawWindowHandle,
        descriptor: &AdapterDescriptor,
    ) -> Result<Self> {
        let instance = Arc::new(AshContext::new(handle, descriptor)?);
        Ok(Self(instance))
    }

    /// Creates a new `Device` from this `Adapter`.
    pub fn request_device(&self, descriptor: &DeviceDescriptor) -> Result<Device> {
        #[cfg(feature = "tracing")]
        {
            let physical_devices = unsafe { self.0.instance.enumerate_physical_devices()? };

            let mut chosen = None;
            self.find_physical_device(descriptor.device_type, physical_devices, &mut chosen);

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
                    self.create_logical_device(physical_device, &descriptor.queue_priority)?;

                info!("Created logical device and queues");

                self.log_surface_info(physical_device)?;

                debug!("Created the default memory allocator");

                Ok(Device {
                    _context: self.0.clone(),
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
            let phys_devs = unsafe { self.0.instance.enumerate_physical_devices()? };

            let mut chosen = None;
            self.find_physical_device(descriptor.device_type, phys_devs, &mut chosen);

            if let Some((physical_device, _physical_device_properties)) = chosen {
                let (logical_device, (graphics_queue, transfer_queue, compute_queue)) =
                    self.create_logical_device(physical_device, &descriptor.queue_priority)?;

                let allocator = allocator::Allocator::new(
                    self.0.instance.clone(),
                    logical_device.clone(),
                    physical_device,
                );

                Ok(Device {
                    _context: self.0.clone(),
                    logical_device,
                    _graphics_queue: graphics_queue,
                    _transfer_queue: transfer_queue,
                    _compute_queue: compute_queue,
                    allocator,
                })
            } else {
                Err(AscheError::RequestDeviceError)
            }
        }
    }

    #[cfg(feature = "tracing")]
    fn log_surface_info(&self, physical_device: vk::PhysicalDevice) -> Result<()> {
        let capabilities = unsafe {
            self.0
                .surface
                .get_physical_device_surface_capabilities(physical_device, self.0.surface_khr)
        }?;
        let present_modes = unsafe {
            self.0
                .surface
                .get_physical_device_surface_present_modes(physical_device, self.0.surface_khr)
        }?;
        let formats = unsafe {
            self.0
                .surface
                .get_physical_device_surface_formats(physical_device, self.0.surface_khr)
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

        Ok(())
    }

    fn find_physical_device(
        &self,
        device_type: vk::PhysicalDeviceType,
        physical_devices: Vec<vk::PhysicalDevice>,
        chosen: &mut Option<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)>,
    ) {
        for device in physical_devices {
            let properties = unsafe { self.0.instance.get_physical_device_properties(device) };

            if properties.device_type == device_type {
                *chosen = Some((device, properties))
            }
        }
    }

    /// Creates a new logical device. Returns the logical device and three separate queues with the
    /// types vk::QueueFlags::GRAPHICS, vk::QueueFlags::TRANSFER and vk::QueueFlags::COMPUTE.
    fn create_logical_device(
        &self,
        physical_device: vk::PhysicalDevice,
        priorities: &QueuePriorityDescriptor,
    ) -> Result<(ash::Device, (vk::Queue, vk::Queue, vk::Queue))> {
        let queue_family_properties = unsafe {
            self.0
                .instance
                .get_physical_device_queue_family_properties(physical_device)
        };

        let graphics_queue_family_id = self.find_queue_family(
            physical_device,
            vk::QueueFlags::GRAPHICS,
            &queue_family_properties,
        )?;
        let transfer_queue_family_id = self.find_queue_family(
            physical_device,
            vk::QueueFlags::TRANSFER,
            &queue_family_properties,
        )?;
        let compute_queue_family_id = self.find_queue_family(
            physical_device,
            vk::QueueFlags::COMPUTE,
            &queue_family_properties,
        )?;

        self.create_queues_and_device(
            physical_device,
            priorities,
            graphics_queue_family_id,
            transfer_queue_family_id,
            compute_queue_family_id,
        )
    }

    fn create_queues_and_device(
        &self,
        physical_device: vk::PhysicalDevice,
        priorities: &QueuePriorityDescriptor,
        graphics_queue_family_id: u32,
        transfer_queue_family_id: u32,
        compute_queue_family_id: u32,
    ) -> Result<(ash::Device, (vk::Queue, vk::Queue, vk::Queue))> {
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
            let logical_device = self.create_device(physical_device, &queue_infos)?;
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
                    .queue_priorities(&[priorities.graphics])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_family_id)
                    .queue_priorities(&[priorities.transfer, priorities.compute])
                    .build(),
            ];
            let logical_device = self.create_device(physical_device, &queue_infos)?;
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
                    .queue_priorities(&[priorities.graphics, priorities.compute])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_family_id)
                    .queue_priorities(&[priorities.transfer])
                    .build(),
            ];
            let logical_device = self.create_device(physical_device, &queue_infos)?;
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
                .queue_priorities(&[priorities.graphics, priorities.transfer, priorities.compute])
                .build()];
            let logical_device = self.create_device(physical_device, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 1) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 2) };

            Ok((logical_device, (g_q, t_q, c_q)))
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
            let logical_device = self.create_device(physical_device, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 0) };

            Ok((logical_device, (g_q, t_q, c_q)))
        }
    }

    fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_infos: &[vk::DeviceQueueCreateInfo],
    ) -> Result<ash::Device> {
        let device_extensions = self.create_device_extensions(physical_device)?;

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

        let layer_pointers = self
            .0
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
            self.0
                .instance
                .create_device(physical_device, &device_create_info, None)?
        };

        Ok(logical_device)
    }

    fn create_device_extensions(&self, physical_device: vk::PhysicalDevice) -> Result<Vec<&CStr>> {
        let mut extensions: Vec<&'static CStr> = Vec::new();

        extensions.push(ash::extensions::khr::Swapchain::name());

        // Only keep available extensions.
        let device_extensions = unsafe {
            self.0
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
        &self,
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
                            self.0.surface.get_physical_device_surface_support(
                                physical_device,
                                id as u32,
                                self.0.surface_khr,
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

/// Wraps some ash structures globally used.
struct AshContext {
    _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) surface_khr: vk::SurfaceKHR,
    pub(crate) surface: ash::extensions::khr::Surface,
    layers: Vec<&'static CStr>,
    #[cfg(feature = "tracing")]
    debug_messenger: Option<DebugMessenger>,
}

#[cfg(feature = "tracing")]
struct DebugMessenger {
    pub(crate) ext: ext::DebugUtils,
    pub(crate) callback: vk::DebugUtilsMessengerEXT,
}

impl AshContext {
    /// Creates a new `Instance`.
    pub fn new(
        handle: &raw_window_handle::RawWindowHandle,
        descriptor: &AdapterDescriptor,
    ) -> Result<Self> {
        let entry = ash::Entry::new()?;

        let engine_name = std::ffi::CString::new("asche")?;
        let app_name = std::ffi::CString::new(descriptor.app_name.clone())?;

        info!("Requesting Vulkan API version: 1.2",);

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(descriptor.app_version)
            .engine_name(&engine_name)
            .engine_version(vk::make_version(0, 1, 0))
            .api_version(vk::make_version(1, 2, 0));

        // Activate all needed instance layers and extensions.
        let instance_extensions = entry
            .enumerate_instance_extension_properties()
            .map_err(|e| {
                #[cfg(feature = "tracing")]
                error!("Unable to enumerate instance extensions: {:?}", e);
                AscheError::Unspecified(format!("Unable to enumerate instance extensions: {:?}", e))
            })?;

        let instance_layers = entry.enumerate_instance_layer_properties().map_err(|e| {
            #[cfg(feature = "tracing")]
            error!("Unable to enumerate instance layers: {:?}", e);
            AscheError::Unspecified(format!("Unable to enumerate instance layers: {:?}", e))
        })?;

        let extensions = Self::create_instance_extensions(&instance_extensions);
        let layers = Self::create_layers(instance_layers);
        let instance = Self::create_instance(&entry, &app_info, &extensions, &layers)?;
        let (surface_khr, surface) = Self::create_surface(&entry, &instance, &handle)?;

        #[cfg(feature = "tracing")]
        {
            let debug_messenger =
                Self::create_debug_messenger(&entry, instance_extensions, &instance);

            Ok(Self {
                _entry: entry,
                instance,
                layers,
                surface_khr,
                surface,
                debug_messenger,
            })
        }
        #[cfg(not(feature = "tracing"))]
        {
            Ok(Self {
                _entry: entry,
                instance: instance,
                layers,
                surface_khr,
                surface,
            })
        }
    }

    #[cfg(feature = "tracing")]
    fn create_debug_messenger(
        entry: &ash::Entry,
        instance_extensions: Vec<vk::ExtensionProperties>,
        instance: &ash::Instance,
    ) -> Option<DebugMessenger> {
        let debug_messenger = {
            let debug_utils = instance_extensions.iter().any(|props| unsafe {
                CStr::from_ptr(props.extension_name.as_ptr()) == ext::DebugUtils::name()
            });

            if debug_utils {
                let ext = ext::DebugUtils::new(entry, instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                    .message_severity(Self::vulkan_log_level())
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                    .pfn_user_callback(Some(debug::debug_utils_callback));
                let callback = unsafe { ext.create_debug_utils_messenger(&info, None) }.unwrap();
                Some(DebugMessenger { ext, callback })
            } else {
                warn!("Unable to create Debug Utils");
                None
            }
        };
        debug_messenger
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

    fn create_instance(
        entry: &ash::Entry,
        app_info: &vk::ApplicationInfoBuilder,
        extensions: &[&CStr],
        layers: &[&CStr],
    ) -> Result<ash::Instance> {
        let str_pointers = layers
            .iter()
            .chain(extensions.iter())
            .map(|&s| {
                // Safe because `layers` and `extensions` entries have static lifetime.
                s.as_ptr()
            })
            .collect::<Vec<_>>();

        #[cfg(feature = "tracing")]
        {
            for extension in extensions.iter() {
                info!("Loading instance extension: {:?}", extension);
            }
        }

        let create_info = vk::InstanceCreateInfo::builder()
            .flags(vk::InstanceCreateFlags::empty())
            .application_info(&app_info)
            .enabled_layer_names(&str_pointers[..layers.len()])
            .enabled_extension_names(&str_pointers[layers.len()..]);

        Ok(
            unsafe { entry.create_instance(&create_info, None) }.map_err(|e| {
                #[cfg(feature = "tracing")]
                error!("Unable to create Vulkan instance: {:?}", e);
                AscheError::Unspecified(format!("Unable to create Vulkan instance: {:?}", e))
            })?,
        )
    }

    fn create_layers(instance_layers: Vec<vk::LayerProperties>) -> Vec<&'static CStr> {
        let mut layers: Vec<&'static CStr> = Vec::new();
        if cfg!(debug_assertions) {
            layers.push(CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());
        }

        // Only keep available layers.
        layers.retain(|&layer| {
            let found = instance_layers.iter().any(|inst_layer| unsafe {
                CStr::from_ptr(inst_layer.layer_name.as_ptr()) == layer
            });
            if found {
                true
            } else {
                #[cfg(feature = "tracing")]
                warn!("Unable to find layer: {}", layer.to_string_lossy());
                false
            }
        });

        layers
    }

    fn create_instance_extensions(instance_extensions: &[vk::ExtensionProperties]) -> Vec<&CStr> {
        let mut extensions: Vec<&'static CStr> = Vec::new();
        extensions.push(ash::extensions::khr::Surface::name());

        // Platform-specific WSI extensions
        if cfg!(all(
            unix,
            not(target_os = "android"),
            not(target_os = "macos")
        )) {
            extensions.push(ash::extensions::khr::XlibSurface::name());
            extensions.push(ash::extensions::khr::XcbSurface::name());
            extensions.push(ash::extensions::khr::WaylandSurface::name());
        }
        if cfg!(target_os = "android") {
            extensions.push(ash::extensions::khr::AndroidSurface::name());
        }
        if cfg!(target_os = "windows") {
            extensions.push(ash::extensions::khr::Win32Surface::name());
        }
        if cfg!(target_os = "macos") {
            extensions.push(ash::extensions::mvk::MacOSSurface::name());
        }

        extensions.push(vk::KhrGetPhysicalDeviceProperties2Fn::name());

        #[cfg(feature = "tracing")]
        {
            extensions.push(ext::DebugUtils::name());
        }

        // Only keep available extensions.
        extensions.retain(|&ext| {
            let found = instance_extensions
                .iter()
                .any(|inst_ext| unsafe { CStr::from_ptr(inst_ext.extension_name.as_ptr()) == ext });
            if found {
                true
            } else {
                #[cfg(feature = "tracing")]
                error!(
                    "Unable to find instance extension: {}",
                    ext.to_string_lossy()
                );
                false
            }
        });

        extensions
    }

    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        handle: &RawWindowHandle,
    ) -> Result<(vk::SurfaceKHR, ash::extensions::khr::Surface)> {
        match handle {
            #[cfg(windows)]
            RawWindowHandle::Windows(h) => {
                let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                    .hwnd(h.hwnd)
                    .hinstance(h.hinstance);

                let win32_surface = ash::extensions::khr::Win32Surface::new(entry, instance);
                let surface_khr =
                    unsafe { win32_surface.create_win32_surface(&create_info, None) }?;
                let surface = ash::extensions::khr::Surface::new(entry, instance);
                Ok((surface_khr, surface))
            }
            #[cfg(unix)]
            RawWindowHandle::Xlib(h) => {
                let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
                    .window(h.window)
                    .dpy(h.display as *mut vk::Display);

                let xlib_surface = ash::extensions::khr::XlibSurface::new(entry, instance);
                let surface_khr = unsafe { xlib_surface.create_xlib_surface(&create_info, None) }?;
                let surface = ash::extensions::khr::Surface::new(entry, instance);
                Ok((surface_khr, surface))
            }
            #[cfg(unix)]
            RawWindowHandle::Xcb(h) => {
                let create_info = vk::XcbSurfaceCreateInfoKHR::builder()
                    .window(h.window)
                    .connection(h.connection);

                let xcb_surface = ash::extensions::khr::XcbSurface::new(entry, instance);
                let surface_khr = unsafe { xcb_surface.create_xcb_surface(&create_info, None) }?;
                let surface = ash::extensions::khr::Surface::new(entry, instance);
                Ok((surface_khr, surface))
            }
            #[cfg(unix)]
            RawWindowHandle::Wayland(h) => {
                let create_info = vk::WaylandSurfaceCreateInfoKHR::builder()
                    .surface(h.surface)
                    .display(h.display);

                let wayland_surface = ash::extensions::khr::WaylandSurface::new(entry, instance);
                let surface_khr =
                    unsafe { wayland_surface.create_wayland_surface(&create_info, None) }?;
                let surface = ash::extensions::khr::Surface::new(entry, instance);
                Ok((surface_khr, surface))
            }
            _ => {
                panic!("Surface creation only supported for windows and unix")
            }
        }
    }
}

impl Drop for AshContext {
    fn drop(&mut self) {
        unsafe {
            #[cfg(feature = "tracing")]
            if let Some(messenger) = &self.debug_messenger {
                messenger
                    .ext
                    .destroy_debug_utils_messenger(messenger.callback, None);
            }

            self.surface.destroy_surface(self.surface_khr, None);
            self.instance.destroy_instance(None);
        };
    }
}

/// Abstracts a Vulkan device.
pub struct Device {
    _context: Arc<AshContext>,
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
