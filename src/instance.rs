use std::ffi::CStr;

use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
use raw_window_handle::RawWindowHandle;
#[cfg(feature = "tracing")]
use tracing::{error, info, level_filters::LevelFilter, warn};

use crate::{
    AscheError, ComputeQueue, Device, DeviceConfiguration, GraphicsQueue, Result, TransferQueue,
};

/// Describes how the instance should be configured.
pub struct InstanceConfiguration<'a> {
    /// Name of the application.
    pub app_name: &'a str,
    /// Version of the application. Use `ash::vk::make_version()` to create the version number.
    pub app_version: u32,
    /// Raw window handle.
    pub handle: &'a raw_window_handle::RawWindowHandle,
    /// Instance extensions to load.
    pub extensions: Vec<&'static CStr>,
}

/// Initializes the all Vulkan resources needed to create a device.
pub struct Instance {
    _entry: ash::Entry,
    /// The raw Vulkan instance.
    pub raw: ash::Instance,
    pub(crate) surface_loader: ash::extensions::khr::Surface,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) layers: Vec<&'static CStr>,
    pub(crate) debug_utils: ash::extensions::ext::DebugUtils,
    pub(crate) debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl Instance {
    /// Creates a new `Instance`.
    pub fn new(configuration: InstanceConfiguration) -> Result<Instance> {
        let entry = ash::Entry::new()?;

        let engine_name = std::ffi::CString::new("asche")?;
        let app_name = std::ffi::CString::new(configuration.app_name.to_owned())?;

        #[cfg(feature = "tracing")]
        info!("Requesting Vulkan API version: 1.2");

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(configuration.app_version)
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

        let extensions = Self::create_instance_extensions(&configuration, &instance_extensions);
        let layers = Self::create_layers(instance_layers);
        let instance = Self::create_instance(&entry, &app_info, &extensions, &layers)?;
        let (surface, surface_loader) =
            Self::create_surface(&entry, &instance, configuration.handle)?;

        let (debug_utils, debug_messenger) =
            Self::create_debug_utils(&entry, instance_extensions, &instance)?;

        Ok(Self {
            _entry: entry,
            raw: instance,
            layers,
            surface,
            surface_loader,
            debug_utils,
            debug_messenger,
        })
    }

    /// Requests a new Vulkan device.
    pub fn request_device(
        self,
        device_configuration: DeviceConfiguration,
    ) -> Result<(Device, (ComputeQueue, GraphicsQueue, TransferQueue))> {
        Device::new(self, device_configuration)
    }

    fn create_debug_utils(
        entry: &ash::Entry,
        instance_extensions: Vec<vk::ExtensionProperties>,
        instance: &ash::Instance,
    ) -> Result<(ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT)> {
        let debug_utils_found = instance_extensions.iter().any(|props| unsafe {
            CStr::from_ptr(props.extension_name.as_ptr())
                == ash::extensions::ext::DebugUtils::name()
        });

        if debug_utils_found {
            let ext = ash::extensions::ext::DebugUtils::new(entry, instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(Self::vulkan_log_level())
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(crate::debug_utils_callback));

            let callback = unsafe { ext.create_debug_utils_messenger(&info, None) }.unwrap();
            Ok((ext, callback))
        } else {
            Err(AscheError::DebugUtilsMissing)
        }
    }

    fn vulkan_log_level() -> vk::DebugUtilsMessageSeverityFlagsEXT {
        #[cfg(feature = "tracing")]
        {
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
        #[cfg(not(feature = "tracing"))]
        {
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
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

    fn create_instance_extensions(
        configuration: &InstanceConfiguration,
        instance_extensions: &[vk::ExtensionProperties],
    ) -> Vec<&'static CStr> {
        let mut extensions: Vec<&'static CStr> = configuration.extensions.clone();
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
        extensions.push(ash::extensions::ext::DebugUtils::name());

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
                let surface = unsafe { win32_surface.create_win32_surface(&create_info, None) }?;
                let surface_loader = ash::extensions::khr::Surface::new(entry, instance);
                Ok((surface, surface_loader))
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

    pub(crate) fn find_physical_device(
        &self,
        device_type: vk::PhysicalDeviceType,
    ) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> {
        let physical_devices = unsafe { self.raw.enumerate_physical_devices()? };

        let mut chosen: Option<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> = None;
        for device in physical_devices {
            let properties = unsafe { self.raw.get_physical_device_properties(device) };

            if properties.device_type == device_type {
                chosen = Some((device, properties))
            }
        }
        let chosen = chosen.ok_or(AscheError::RequestDeviceError)?;

        Ok(chosen)
    }

    /// Creates a new logical device. Returns the logical device, and the queue families and the Vulkan queues.
    /// We have three queues in the following order:
    ///  * Compute queue
    ///  * Graphics queue
    ///  * Transfer queue
    ///
    /// The compute and graphics queue use dedicated queue families if provided by the implementation.
    /// The graphics queue is guaranteed to be able to write the the surface.
    pub(crate) fn create_logical_device(
        &self,
        physical_device: vk::PhysicalDevice,
        configuration: DeviceConfiguration,
    ) -> Result<(ash::Device, [u32; 3], [vk::Queue; 3])> {
        let queue_family_properties = unsafe {
            self.raw
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

        self.create_logical_device_and_queues(
            physical_device,
            configuration,
            graphics_queue_family_id,
            transfer_queue_family_id,
            compute_queue_family_id,
        )
    }

    fn create_logical_device_and_queues(
        &self,
        physical_device: vk::PhysicalDevice,
        configuration: DeviceConfiguration,
        graphics_queue_family_id: u32,
        transfer_queue_family_id: u32,
        compute_queue_family_id: u32,
    ) -> Result<(ash::Device, [u32; 3], [vk::Queue; 3])> {
        let priorities = &configuration.queue_priority;

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
            let logical_device =
                self.create_device(physical_device, configuration, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 1) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 0) };

            Ok((
                logical_device,
                [
                    compute_queue_family_id,
                    graphics_queue_family_id,
                    transfer_queue_family_id,
                ],
                [c_q, g_q, t_q],
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
            let logical_device =
                self.create_device(physical_device, configuration, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 1) };

            Ok((
                logical_device,
                [
                    compute_queue_family_id,
                    graphics_queue_family_id,
                    transfer_queue_family_id,
                ],
                [c_q, g_q, t_q],
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
            let logical_device =
                self.create_device(physical_device, configuration, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 1) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };

            Ok((
                logical_device,
                [
                    compute_queue_family_id,
                    graphics_queue_family_id,
                    transfer_queue_family_id,
                ],
                [c_q, g_q, t_q],
            ))
        } else if graphics_queue_family_id == transfer_queue_family_id
            && transfer_queue_family_id == compute_queue_family_id
        {
            // Case: G=T=C
            let queue_infos = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_family_id)
                .queue_priorities(&[priorities.graphics, priorities.transfer, priorities.compute])
                .build()];
            let logical_device =
                self.create_device(physical_device, configuration, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 1) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 2) };

            Ok((
                logical_device,
                [
                    compute_queue_family_id,
                    graphics_queue_family_id,
                    transfer_queue_family_id,
                ],
                [c_q, g_q, t_q],
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
            let logical_device =
                self.create_device(physical_device, configuration, &queue_infos)?;
            let g_q = unsafe { logical_device.get_device_queue(graphics_queue_family_id, 0) };
            let t_q = unsafe { logical_device.get_device_queue(transfer_queue_family_id, 0) };
            let c_q = unsafe { logical_device.get_device_queue(compute_queue_family_id, 0) };

            Ok((
                logical_device,
                [
                    compute_queue_family_id,
                    graphics_queue_family_id,
                    transfer_queue_family_id,
                ],
                [c_q, g_q, t_q],
            ))
        }
    }

    fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        mut configuration: DeviceConfiguration,
        queue_infos: &[vk::DeviceQueueCreateInfo],
    ) -> Result<ash::Device> {
        let device_extensions = self.create_device_extensions(physical_device, &configuration)?;

        #[cfg(feature = "tracing")]
        {
            for extension in device_extensions.iter() {
                info!("Loading device extension: {:?}", extension);
            }
        }

        let layer_pointers = self.layers.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();

        let extension_pointers = device_extensions
            .iter()
            .map(|&s| s.as_ptr())
            .collect::<Vec<_>>();

        let features = if let Some(features) = configuration.features_v1_0.take() {
            features
        } else {
            vk::PhysicalDeviceFeatures::builder()
        };
        let mut features11 = if let Some(features) = configuration.features_v1_1.take() {
            features
        } else {
            vk::PhysicalDeviceVulkan11Features::builder()
        };
        let mut features12 = if let Some(features) = configuration.features_v1_2.take() {
            features
        } else {
            vk::PhysicalDeviceVulkan12Features::builder()
        };

        features12 = features12.timeline_semaphore(true);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_infos)
            .enabled_extension_names(&extension_pointers)
            .enabled_layer_names(&layer_pointers)
            .enabled_features(&features)
            .push_next(&mut features11)
            .push_next(&mut features12);

        let logical_device = unsafe {
            self.raw
                .create_device(physical_device, &device_create_info, None)?
        };

        Ok(logical_device)
    }

    fn create_device_extensions(
        &self,
        physical_device: vk::PhysicalDevice,
        configuration: &DeviceConfiguration,
    ) -> Result<Vec<&CStr>> {
        let mut extensions: Vec<&'static CStr> = configuration.extensions.clone();

        extensions.push(ash::extensions::khr::Swapchain::name());

        // Only keep available extensions.
        let device_extensions = unsafe {
            self.raw
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
                            self.surface_loader.get_physical_device_surface_support(
                                physical_device,
                                id as u32,
                                self.surface,
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

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);

            self.surface_loader.destroy_surface(self.surface, None);
            self.raw.destroy_instance(None);
        };
    }
}
