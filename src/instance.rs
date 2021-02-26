use std::ffi::CStr;

use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
use ash::vk;
use raw_window_handle::RawWindowHandle;
#[cfg(feature = "tracing")]
use tracing::{error, info, warn};

#[cfg(debug_assertions)]
use crate::vk_debug::debug_utils_callback;
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
    layers: Vec<&'static CStr>,
    #[cfg(debug_assertions)]
    pub(crate) debug_utils: ash::extensions::ext::DebugUtils,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
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

        #[cfg(debug_assertions)]
        let (debug_utils, debug_messenger) =
            Self::create_debug_utils(&entry, instance_extensions, &instance)?;

        Ok(Self {
            _entry: entry,
            raw: instance,
            layers,
            surface,
            surface_loader,
            #[cfg(debug_assertions)]
            debug_utils,
            #[cfg(debug_assertions)]
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

    #[cfg(debug_assertions)]
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
                .pfn_user_callback(Some(debug_utils_callback));

            let callback = unsafe { ext.create_debug_utils_messenger(&info, None) }.unwrap();
            Ok((ext, callback))
        } else {
            Err(AscheError::DebugUtilsMissing)
        }
    }

    #[cfg(debug_assertions)]
    fn vulkan_log_level() -> vk::DebugUtilsMessageSeverityFlagsEXT {
        #[cfg(feature = "tracing")]
        {
            use tracing::level_filters::LevelFilter;
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
            info!("Loading instance extensions:");
            for extension in extensions.iter() {
                info!("- {:?}", extension);
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

        #[cfg(debug_assertions)]
        layers.push(CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());

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

        #[cfg(debug_assertions)]
        extensions.push(ash::extensions::ext::DebugUtils::name());

        // Only keep available extensions.
        extensions.retain(|&ext| {
            instance_extensions
                .iter()
                .any(|inst_ext| unsafe { CStr::from_ptr(inst_ext.extension_name.as_ptr()) == ext })
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
    ) -> Result<(
        vk::PhysicalDevice,
        vk::PhysicalDeviceProperties,
        vk::PhysicalDeviceDriverProperties,
    )> {
        let physical_devices = unsafe { self.raw.enumerate_physical_devices()? };

        // We try to find our preferred device type.
        let mut chosen = self.find_physical_device_inner_loop(device_type, &physical_devices);

        // We try to fall back to the next best thing.
        if chosen.is_none() {
            if device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                chosen = self.find_physical_device_inner_loop(
                    vk::PhysicalDeviceType::INTEGRATED_GPU,
                    &physical_devices,
                );
            }
            if device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                chosen = self.find_physical_device_inner_loop(
                    vk::PhysicalDeviceType::DISCRETE_GPU,
                    &physical_devices,
                );
            }
        }

        let chosen = chosen.ok_or(AscheError::RequestDeviceError)?;

        Ok(chosen)
    }

    fn find_physical_device_inner_loop(
        &self,
        device_type: vk::PhysicalDeviceType,
        physical_devices: &[vk::PhysicalDevice],
    ) -> Option<(
        vk::PhysicalDevice,
        vk::PhysicalDeviceProperties,
        vk::PhysicalDeviceDriverProperties,
    )> {
        let mut chosen = None;
        for device in physical_devices {
            let mut physical_device_driver_properties =
                vk::PhysicalDeviceDriverProperties::default();
            let mut physical_device_properties = vk::PhysicalDeviceProperties2::builder()
                .push_next(&mut physical_device_driver_properties);

            unsafe {
                self.raw
                    .get_physical_device_properties2(*device, &mut physical_device_properties)
            };

            if physical_device_properties.properties.device_type == device_type {
                chosen = Some((
                    *device,
                    physical_device_properties.properties,
                    physical_device_driver_properties,
                ))
            }
        }
        chosen
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
            info!("Loading device extensions:");
            for extension in device_extensions.iter() {
                info!("- {:?}", extension);
            }
        }

        let layer_pointers = self.layers.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();

        let extension_pointers = device_extensions
            .iter()
            .map(|&s| s.as_ptr())
            .collect::<Vec<_>>();

        let (physical_features, physical_features11, physical_features12) =
            self.collect_physical_device_features(physical_device);

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

        features12 = features12
            .buffer_device_address(true)
            .timeline_semaphore(true);

        check_features(
            &physical_features.features,
            &physical_features11,
            &physical_features12,
            &features,
            &features11,
            &features12,
        )?;

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

    fn collect_physical_device_features(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> (
        vk::PhysicalDeviceFeatures2Builder,
        vk::PhysicalDeviceVulkan11FeaturesBuilder,
        vk::PhysicalDeviceVulkan12FeaturesBuilder,
    ) {
        let mut physical_features = vk::PhysicalDeviceFeatures2::builder();
        let mut physical_features11 = vk::PhysicalDeviceVulkan11Features::builder();
        let mut physical_features12 = vk::PhysicalDeviceVulkan12Features::builder();
        unsafe {
            // Workaround until this PR is finished and merged: https://github.com/MaikKlein/ash/issues/325
            // Since the vk::PhysicalDeviceFeatures2Builder misses the pushNext() method.
            physical_features.p_next = &mut physical_features11 as *mut _ as *mut std::ffi::c_void;
            physical_features.p_next = &mut physical_features12 as *mut _ as *mut std::ffi::c_void;
            self.raw
                .get_physical_device_features2(physical_device, &mut physical_features)
        };
        (physical_features, physical_features11, physical_features12)
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
            device_extensions
                .iter()
                .any(|inst_ext| unsafe { CStr::from_ptr(inst_ext.extension_name.as_ptr()) == ext })
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
            info!("Selected {:?} queue has family index {}", target_family, id);

            Ok(id)
        } else {
            match target_family {
                vk::QueueFlags::COMPUTE => {
                    Err(AscheError::QueueFamilyNotFound("compute".to_string()))
                }
                vk::QueueFlags::GRAPHICS => {
                    Err(AscheError::QueueFamilyNotFound("graphic".to_string()))
                }
                vk::QueueFlags::TRANSFER => {
                    Err(AscheError::QueueFamilyNotFound("transfer".to_string()))
                }
                _ => panic!("Unhandled vk::QueueFlags value"),
            }
        }
    }
}

macro_rules! impl_feature_assemble {
    (
        $present:ident, $wanted:ident, $present_vector:ident, $wanted_vector:ident;
        {
            $($name:ident)*
        }
    ) => {
        $(if $present.$name != 0 {
            $present_vector.push(stringify!($name))
        })*
        $(if $wanted.$name != 0 {
            $wanted_vector.push(stringify!($name))
        })*
    };
}

fn check_features(
    physical_features: &vk::PhysicalDeviceFeatures,
    physical_features11: &vk::PhysicalDeviceVulkan11FeaturesBuilder,
    physical_features12: &vk::PhysicalDeviceVulkan12FeaturesBuilder,
    features: &vk::PhysicalDeviceFeaturesBuilder,
    features11: &vk::PhysicalDeviceVulkan11FeaturesBuilder,
    features12: &vk::PhysicalDeviceVulkan12FeaturesBuilder,
) -> Result<()> {
    let mut physical_features_list: Vec<&str> = Vec::with_capacity(54);
    let mut physical_features11_list: Vec<&str> = Vec::with_capacity(11);
    let mut physical_features12_list: Vec<&str> = Vec::with_capacity(46);
    let mut features_list: Vec<&str> = Vec::with_capacity(54);
    let mut features11_list: Vec<&str> = Vec::with_capacity(11);
    let mut features12_list: Vec<&str> = Vec::with_capacity(46);

    impl_feature_assemble!(
        physical_features, features, physical_features_list, features_list;
        {
            robust_buffer_access
            full_draw_index_uint32
            image_cube_array
            independent_blend
            geometry_shader
            tessellation_shader
            sample_rate_shading
            dual_src_blend
            logic_op
            multi_draw_indirect
            draw_indirect_first_instance
            depth_clamp
            depth_bias_clamp
            fill_mode_non_solid
            depth_bounds
            wide_lines
            large_points
            alpha_to_one
            multi_viewport
            sampler_anisotropy
            texture_compression_etc2
            texture_compression_astc_ldr
            texture_compression_bc
            occlusion_query_precise
            pipeline_statistics_query
            vertex_pipeline_stores_and_atomics
            fragment_stores_and_atomics
            shader_tessellation_and_geometry_point_size
            shader_image_gather_extended
            shader_storage_image_extended_formats
            shader_storage_image_multisample
            shader_storage_image_read_without_format
            shader_storage_image_write_without_format
            shader_uniform_buffer_array_dynamic_indexing
            shader_sampled_image_array_dynamic_indexing
            shader_storage_buffer_array_dynamic_indexing
            shader_storage_image_array_dynamic_indexing
            shader_clip_distance
            shader_cull_distance
            shader_float64
            shader_int64
            shader_int16
            shader_resource_residency
            shader_resource_min_lod
            sparse_binding
            sparse_residency_buffer
            sparse_residency_image2_d
            sparse_residency_image3_d
            sparse_residency2_samples
            sparse_residency4_samples
            sparse_residency8_samples
            sparse_residency16_samples
            sparse_residency_aliased
            variable_multisample_rate
            inherited_queries
        }
    );

    impl_feature_assemble!(
        physical_features11, features11, physical_features11_list, features11_list;
        {
            storage_buffer16_bit_access
            uniform_and_storage_buffer16_bit_access
            storage_push_constant16
            storage_input_output16
            multiview
            multiview_geometry_shader
            multiview_tessellation_shader
            variable_pointers_storage_buffer
            variable_pointers
            protected_memory
            sampler_ycbcr_conversion
            shader_draw_parameters
        }
    );

    impl_feature_assemble!(
        physical_features12, features12, physical_features12_list, features12_list;
        {
            sampler_mirror_clamp_to_edge
            draw_indirect_count
            storage_buffer8_bit_access
            uniform_and_storage_buffer8_bit_access
            storage_push_constant8
            shader_buffer_int64_atomics
            shader_shared_int64_atomics
            shader_float16
            shader_int8
            descriptor_indexing
            shader_input_attachment_array_dynamic_indexing
            shader_uniform_texel_buffer_array_dynamic_indexing
            shader_storage_texel_buffer_array_dynamic_indexing
            shader_uniform_buffer_array_non_uniform_indexing
            shader_sampled_image_array_non_uniform_indexing
            shader_storage_buffer_array_non_uniform_indexing
            shader_storage_image_array_non_uniform_indexing
            shader_input_attachment_array_non_uniform_indexing
            shader_uniform_texel_buffer_array_non_uniform_indexing
            shader_storage_texel_buffer_array_non_uniform_indexing
            descriptor_binding_uniform_buffer_update_after_bind
            descriptor_binding_sampled_image_update_after_bind
            descriptor_binding_storage_image_update_after_bind
            descriptor_binding_storage_buffer_update_after_bind
            descriptor_binding_uniform_texel_buffer_update_after_bind
            descriptor_binding_storage_texel_buffer_update_after_bind
            descriptor_binding_update_unused_while_pending
            descriptor_binding_partially_bound
            descriptor_binding_variable_descriptor_count
            runtime_descriptor_array
            sampler_filter_minmax
            scalar_block_layout
            imageless_framebuffer
            uniform_buffer_standard_layout
            shader_subgroup_extended_types
            separate_depth_stencil_layouts
            host_query_reset
            timeline_semaphore
            buffer_device_address
            buffer_device_address_capture_replay
            buffer_device_address_multi_device
            vulkan_memory_model
            vulkan_memory_model_device_scope
            vulkan_memory_model_availability_visibility_chains
            shader_output_viewport_index
            shader_output_layer
            subgroup_broadcast_dynamic_id
        }
    );

    let mut missing_features = false;

    #[cfg(feature = "tracing")]
    info!("Enabling Vulkan 1.0 device feature:");
    features_list.retain(|&wanted| {
        let found = physical_features_list
            .iter()
            .any(|present| *present == wanted);
        if found {
            info!("- {}", wanted);
            true
        } else {
            #[cfg(feature = "tracing")]
            error!("Unable to find Vulkan 1.0 feature: {}", wanted);
            missing_features = true;
            false
        }
    });

    #[cfg(feature = "tracing")]
    info!("Enabling Vulkan 1.1 device feature:");
    features11_list.retain(|&wanted| {
        let found = physical_features11_list
            .iter()
            .any(|present| *present == wanted);
        if found {
            info!("- {}", wanted);
            true
        } else {
            #[cfg(feature = "tracing")]
            error!("Unable to find Vulkan 1.1 feature: {}", wanted);
            missing_features = true;
            false
        }
    });

    info!("Enabling Vulkan 1.2 device feature:");
    features12_list.retain(|&wanted| {
        let found = physical_features12_list
            .iter()
            .any(|present| *present == wanted);
        if found {
            info!("- {}", wanted);
            true
        } else {
            #[cfg(feature = "tracing")]
            error!("Unable to find Vulkan 1.2 feature: {}", wanted);
            missing_features = true;
            false
        }
    });

    if missing_features {
        Err(AscheError::SwapchainFormatIncompatible)
    } else {
        Ok(())
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            #[cfg(debug_assertions)]
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);

            self.surface_loader.destroy_surface(self.surface, None);
            self.raw.destroy_instance(None);
        };
    }
}
