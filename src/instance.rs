use std::convert::TryInto;
use std::ffi::CStr;
use std::fmt::Formatter;
use std::os::raw::c_char;

use erupt::{vk, ExtendableFromConst, ExtendableFromMut};
#[cfg(feature = "tracing")]
use tracing1::{error, info, warn};

use crate::device::Queues;
#[cfg(debug_assertions)]
use crate::vk_debug::debug_utils_callback;
use crate::{AscheError, Device, DeviceConfiguration, Lifetime, Result, Swapchain};

/// Describes how the instance should be configured.
#[derive(Clone, Debug)]
pub struct InstanceConfiguration<'a> {
    /// Name of the application.
    pub app_name: &'a str,
    /// Version of the application.
    pub app_version: Version,
    /// Name of the engine.
    pub engine_name: &'a str,
    /// Version of the engine.
    pub engine_version: Version,
    /// Instance extensions to load.
    pub extensions: Vec<*const c_char>,
}

/// A version number.
#[derive(Clone, Debug, Copy, Eq, PartialEq)]
pub struct Version {
    /// The version major.
    pub major: u32,
    /// The version minor.
    pub minor: u32,
    /// The version patch.
    pub patch: u32,
}

impl From<Version> for u32 {
    fn from(v: Version) -> Self {
        vk::make_api_version(0, v.major, v.minor, v.patch)
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Initializes the all Vulkan resources needed to create a device.
#[derive(Debug)]
pub struct Instance {
    pub(crate) surface: vk::SurfaceKHR,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
    raw: erupt::InstanceLoader,
    _entry: erupt::EntryLoader,
}

impl Instance {
    /// Creates a new `Instance`.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub fn new(
        window_handle: &impl raw_window_handle::HasRawWindowHandle,
        configuration: InstanceConfiguration,
    ) -> Result<Instance> {
        let entry = erupt::EntryLoader::new()?;

        let app_name = std::ffi::CString::new(configuration.app_name.to_owned())?;
        let engine_name = std::ffi::CString::new(configuration.engine_name.to_owned())?;

        #[cfg(feature = "tracing")]
        {
            info!("Application name: {}", configuration.app_name);
            info!("Application version: {}", configuration.app_version);
            info!("Engine name: {}", configuration.engine_name);
            info!("Engine version: {}", configuration.engine_version);
            info!("Requesting Vulkan API version: 1.2.0");
        }

        let app_info = vk::ApplicationInfoBuilder::new()
            .application_name(&app_name)
            .application_version(configuration.app_version.into())
            .engine_name(&engine_name)
            .engine_version(configuration.engine_version.into())
            .api_version(vk::make_api_version(0, 1, 2, 0));

        // Activate all needed instance layers and extensions.
        let instance_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None, None) }.map_err(
                |err| {
                    #[cfg(feature = "tracing")]
                    error!("Unable to enumerate instance extensions: {}", err);
                    AscheError::VkResult(err)
                },
            )?;

        let instance_layers =
            unsafe { entry.enumerate_instance_layer_properties(None) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to enumerate instance layers: {}", err);
                AscheError::VkResult(err)
            })?;

        let extensions =
            Self::create_instance_extensions(&configuration, &instance_extensions, window_handle)?;
        let layers = Self::create_layers(&instance_layers);
        let instance = Self::create_instance(&entry, &app_info, &extensions, &layers)?;
        let surface =
            unsafe { erupt::utils::surface::create_surface(&instance, window_handle, None) }
                .map_err(|err| {
                    #[cfg(feature = "tracing")]
                    error!("Unable to create a surface: {}", err);
                    AscheError::VkResult(err)
                })?;

        #[cfg(debug_assertions)]
        let debug_messenger = Self::create_debug_utils(&instance, instance_extensions)?;

        Ok(Self {
            _entry: entry,
            raw: instance,
            surface,
            #[cfg(debug_assertions)]
            debug_messenger,
        })
    }

    /// The raw Vulkan instance handle.
    #[inline]
    pub(crate) fn raw(&self) -> &erupt::InstanceLoader {
        &self.raw
    }

    /// Requests a new Vulkan device. Returns a device, a swapchain and the queues created.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn request_device<LT: Lifetime>(
        self,
        device_configuration: DeviceConfiguration,
    ) -> Result<(Device<LT>, Swapchain, Queues)> {
        Device::new(self, device_configuration)
    }

    #[cfg(debug_assertions)]
    fn create_debug_utils(
        instance: &erupt::InstanceLoader,
        instance_extensions: Vec<vk::ExtensionProperties>,
    ) -> Result<vk::DebugUtilsMessengerEXT> {
        let debug_name = unsafe { CStr::from_ptr(vk::EXT_DEBUG_UTILS_EXTENSION_NAME) };
        let debug_utils_found = instance_extensions
            .iter()
            .any(|props| unsafe { CStr::from_ptr(props.extension_name.as_ptr()) } == debug_name);

        if debug_utils_found {
            let info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
                .message_severity(Self::vulkan_log_level())
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(debug_utils_callback));

            let utils = unsafe { instance.create_debug_utils_messenger_ext(&info, None) }.map_err(
                |err| {
                    #[cfg(feature = "tracing")]
                    error!("Unable to create the debug utils messenger: {}", err);
                    AscheError::VkResult(err)
                },
            )?;
            Ok(utils)
        } else {
            Err(AscheError::DebugUtilsMissing)
        }
    }

    #[cfg(debug_assertions)]
    fn vulkan_log_level() -> vk::DebugUtilsMessageSeverityFlagsEXT {
        #[cfg(feature = "tracing")]
        {
            use tracing1::level_filters::LevelFilter;
            match tracing1::level_filters::STATIC_MAX_LEVEL {
                LevelFilter::OFF => vk::DebugUtilsMessageSeverityFlagsEXT::empty(),
                LevelFilter::ERROR => vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
                LevelFilter::WARN => {
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                }
                LevelFilter::INFO => {
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO_EXT
                }
                LevelFilter::DEBUG => {
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO_EXT
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT
                }
                LevelFilter::TRACE => vk::DebugUtilsMessageSeverityFlagsEXT::all(),
            }
        }
        #[cfg(not(feature = "tracing"))]
        {
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
        }
    }

    fn create_instance(
        entry: &erupt::EntryLoader,
        app_info: &vk::ApplicationInfoBuilder,
        instance_extensions: &[*const c_char],
        layers: &[*const c_char],
    ) -> Result<erupt::InstanceLoader> {
        #[cfg(feature = "tracing")]
        Self::print_extensions("instance", instance_extensions)?;

        let create_info = vk::InstanceCreateInfoBuilder::new()
            .flags(vk::InstanceCreateFlags::empty())
            .application_info(app_info)
            .enabled_layer_names(layers)
            .enabled_extension_names(instance_extensions);

        unsafe { erupt::InstanceLoader::new(entry, &create_info, None) }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create Vulkan instance: {}", err);
            AscheError::LoaderError(err)
        })
    }

    fn create_layers(instance_layers: &[vk::LayerProperties]) -> Vec<*const c_char> {
        let mut layers = Vec::new();

        layers.retain(|layer| {
            let instance_layer = unsafe { CStr::from_ptr(*layer) };
            let found = instance_layers.iter().any(|inst_layer| unsafe {
                CStr::from_ptr(inst_layer.layer_name.as_ptr()) == instance_layer
            });
            if found {
                true
            } else {
                #[cfg(feature = "tracing")]
                warn!("Unable to find layer: {}", instance_layer.to_string_lossy());
                false
            }
        });
        layers
    }

    fn create_instance_extensions(
        configuration: &InstanceConfiguration,
        instance_extensions: &[vk::ExtensionProperties],
        window_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Vec<*const c_char>> {
        let mut extensions: Vec<*const c_char> = configuration.extensions.clone();

        let required_extensions =
            erupt::utils::surface::enumerate_required_extensions(window_handle).map_err(|err| {
                #[cfg(feature = "tracing")]
                error!(
                    "Unable to enumerate the required extensions for the surface creation: {}",
                    err
                );
                AscheError::VkResult(err)
            })?;
        extensions.extend(required_extensions.iter());

        #[cfg(debug_assertions)]
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);

        // Only keep available extensions.
        Instance::retain_extensions(instance_extensions, &mut extensions);

        Ok(extensions)
    }

    fn retain_extensions(
        present_extensions: &[vk::ExtensionProperties],
        requested_extensions: &mut Vec<*const i8>,
    ) {
        requested_extensions.retain(|ext| {
            let extension = unsafe { CStr::from_ptr(*ext) };
            present_extensions.iter().any(|inst_ext| unsafe {
                CStr::from_ptr(inst_ext.extension_name.as_ptr()) == extension
            })
        });
    }

    pub(crate) fn find_physical_device(
        &self,
        device_type: vk::PhysicalDeviceType,
    ) -> Result<(
        vk::PhysicalDevice,
        vk::PhysicalDeviceProperties,
        vk::PhysicalDeviceDriverProperties,
    )> {
        let physical_devices =
            unsafe { self.raw.enumerate_physical_devices(None) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to enumerate the physical devices: {}", err);
                AscheError::VkResult(err)
            })?;

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
            let physical_device_properties = vk::PhysicalDeviceProperties2Builder::new()
                .extend_from(&mut physical_device_driver_properties);

            let physical_device_properties = unsafe {
                self.raw.get_physical_device_properties2(
                    *device,
                    Some(physical_device_properties.build()),
                )
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
    /// We have three queue vectors in the following order:
    ///  * Compute queues
    ///  * Graphics queues
    ///  * Transfer queues
    ///
    /// The compute and graphics queues use dedicated queue families if provided by the implementation.
    /// The graphics queues are guaranteed to be able to write the the surface.
    pub(crate) fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        configuration: DeviceConfiguration,
    ) -> Result<(erupt::DeviceLoader, [u32; 3], [Vec<vk::Queue>; 3])> {
        let queue_family_properties = unsafe {
            self.raw
                .get_physical_device_queue_family_properties(physical_device, None)
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

        self.create_device_and_queues(
            physical_device,
            configuration,
            graphics_queue_family_id,
            transfer_queue_family_id,
            compute_queue_family_id,
        )
    }

    #[allow(clippy::as_conversions)]
    fn create_device_and_queues(
        &self,
        physical_device: vk::PhysicalDevice,
        configuration: DeviceConfiguration,
        graphics_queue_family_id: u32,
        transfer_queue_family_id: u32,
        compute_queue_family_id: u32,
    ) -> Result<(erupt::DeviceLoader, [u32; 3], [Vec<vk::Queue>; 3])> {
        let queue_configuration = &configuration.queue_configuration;

        let graphics_count = queue_configuration.graphics_queues.len();
        let transfer_count = queue_configuration.transfer_queues.len();
        let compute_count = queue_configuration.compute_queues.len();

        if graphics_count + transfer_count + compute_count == 0 {
            return Err(AscheError::NoQueueConfigured);
        }

        if graphics_count > 64 {
            return Err(AscheError::QueueCountTooHigh(graphics_count));
        }

        if transfer_count > 64 {
            return Err(AscheError::QueueCountTooHigh(transfer_count));
        }

        if compute_count > 64 {
            return Err(AscheError::QueueCountTooHigh(compute_count));
        }

        // If some queue families point to the same ID, we need to create only one
        // `vk::DeviceQueueCreateInfo` for them.
        if graphics_queue_family_id == transfer_queue_family_id
            && transfer_queue_family_id != compute_queue_family_id
        {
            // Case: G=T,C
            let mut p1 = Vec::with_capacity(graphics_count + transfer_count);
            p1.extend_from_slice(&queue_configuration.graphics_queues);
            p1.extend_from_slice(&queue_configuration.transfer_queues);

            let mut p2 = Vec::with_capacity(compute_count);
            p2.extend_from_slice(&queue_configuration.compute_queues);

            let mut queue_infos = Vec::with_capacity(2);
            if !p1.is_empty() {
                queue_infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .queue_family_index(graphics_queue_family_id)
                        .queue_priorities(&p1),
                );
            }

            if !p2.is_empty() {
                queue_infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .queue_family_index(compute_queue_family_id)
                        .queue_priorities(&p2),
                );
            }

            let device =
                self.create_logical_device(physical_device, configuration, &queue_infos)?;

            let g_q = (0..graphics_count)
                .map(|i| unsafe { device.get_device_queue(graphics_queue_family_id, i as u32) })
                .collect::<_>();

            let t_q = (graphics_count..graphics_count + transfer_count)
                .map(|i| unsafe { device.get_device_queue(graphics_queue_family_id, i as u32) })
                .collect::<_>();

            let c_q = (0..compute_count)
                .map(|i| unsafe { device.get_device_queue(compute_queue_family_id, i as u32) })
                .collect::<_>();

            Ok((
                device,
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
            let mut p1 = Vec::with_capacity(graphics_count);
            p1.extend_from_slice(&queue_configuration.graphics_queues);

            let mut p2 = Vec::with_capacity(transfer_count + compute_count);
            p2.extend_from_slice(&queue_configuration.transfer_queues);
            p2.extend_from_slice(&queue_configuration.compute_queues);

            let mut queue_infos = Vec::with_capacity(2);
            if !p1.is_empty() {
                queue_infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .queue_family_index(graphics_queue_family_id)
                        .queue_priorities(&p1),
                );
            }
            if !p2.is_empty() {
                queue_infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .queue_family_index(transfer_queue_family_id)
                        .queue_priorities(&p2),
                );
            }

            let device =
                self.create_logical_device(physical_device, configuration, &queue_infos)?;
            let g_q = (0..graphics_count)
                .map(|i| unsafe { device.get_device_queue(graphics_queue_family_id, i as u32) })
                .collect::<_>();

            let t_q = (0..transfer_count)
                .map(|i| unsafe { device.get_device_queue(transfer_queue_family_id, i as u32) })
                .collect::<_>();

            let c_q = (transfer_count..transfer_count + compute_count)
                .map(|i| unsafe { device.get_device_queue(transfer_queue_family_id, i as u32) })
                .collect::<_>();

            Ok((
                device,
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
            let mut p1 = Vec::with_capacity(graphics_count + compute_count);
            p1.extend_from_slice(&queue_configuration.graphics_queues);
            p1.extend_from_slice(&queue_configuration.compute_queues);

            let mut p2 = Vec::with_capacity(transfer_count);
            p2.extend_from_slice(&queue_configuration.transfer_queues);

            let mut queue_infos = Vec::with_capacity(2);
            if !p1.is_empty() {
                queue_infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .queue_family_index(graphics_queue_family_id)
                        .queue_priorities(&p1),
                );
            }
            if !p2.is_empty() {
                queue_infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .queue_family_index(transfer_queue_family_id)
                        .queue_priorities(&p2),
                );
            }

            let device =
                self.create_logical_device(physical_device, configuration, &queue_infos)?;

            let g_q = (0..graphics_count)
                .map(|i| unsafe { device.get_device_queue(graphics_queue_family_id, i as u32) })
                .collect::<_>();

            let c_q = (graphics_count..graphics_count + compute_count)
                .map(|i| unsafe { device.get_device_queue(graphics_queue_family_id, i as u32) })
                .collect::<_>();

            let t_q = (0..transfer_count)
                .map(|i| unsafe { device.get_device_queue(transfer_queue_family_id, i as u32) })
                .collect::<_>();

            Ok((
                device,
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
            let mut p1 = Vec::with_capacity(graphics_count + transfer_count + compute_count);
            p1.extend_from_slice(&queue_configuration.graphics_queues);
            p1.extend_from_slice(&queue_configuration.transfer_queues);
            p1.extend_from_slice(&queue_configuration.compute_queues);

            let queue_infos = [vk::DeviceQueueCreateInfoBuilder::new()
                .queue_family_index(graphics_queue_family_id)
                .queue_priorities(&p1)];
            let device =
                self.create_logical_device(physical_device, configuration, &queue_infos)?;
            let g_q = (0..graphics_count)
                .map(|i| unsafe { device.get_device_queue(graphics_queue_family_id, i as u32) })
                .collect::<_>();

            let t_q = (graphics_count..graphics_count + transfer_count)
                .map(|i| unsafe { device.get_device_queue(graphics_queue_family_id, i as u32) })
                .collect::<_>();

            let c_q = (graphics_count + transfer_count
                ..graphics_count + transfer_count + compute_count)
                .map(|i| unsafe { device.get_device_queue(graphics_queue_family_id, i as u32) })
                .collect::<_>();

            Ok((
                device,
                [
                    compute_queue_family_id,
                    graphics_queue_family_id,
                    transfer_queue_family_id,
                ],
                [c_q, g_q, t_q],
            ))
        } else {
            // Case: G,T,C
            let mut p1 = Vec::with_capacity(graphics_count);
            p1.extend_from_slice(&queue_configuration.graphics_queues);

            let mut p2 = Vec::with_capacity(transfer_count);
            p2.extend_from_slice(&queue_configuration.transfer_queues);

            let mut p3 = Vec::with_capacity(compute_count);
            p3.extend_from_slice(&queue_configuration.compute_queues);

            let mut queue_infos = Vec::with_capacity(2);
            if !p1.is_empty() {
                queue_infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .queue_family_index(graphics_queue_family_id)
                        .queue_priorities(&p1),
                );
            }
            if !p2.is_empty() {
                queue_infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .queue_family_index(transfer_queue_family_id)
                        .queue_priorities(&p2),
                );
            }
            if !p3.is_empty() {
                queue_infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .queue_family_index(compute_queue_family_id)
                        .queue_priorities(&p3),
                );
            }

            let device =
                self.create_logical_device(physical_device, configuration, &queue_infos)?;

            let g_q = (0..graphics_count)
                .map(|i| unsafe { device.get_device_queue(graphics_queue_family_id, i as u32) })
                .collect::<_>();

            let t_q = (0..transfer_count)
                .map(|i| unsafe { device.get_device_queue(transfer_queue_family_id, i as u32) })
                .collect::<_>();

            let c_q = (0..compute_count)
                .map(|i| unsafe { device.get_device_queue(compute_queue_family_id, i as u32) })
                .collect::<_>();

            Ok((
                device,
                [
                    compute_queue_family_id,
                    graphics_queue_family_id,
                    transfer_queue_family_id,
                ],
                [c_q, g_q, t_q],
            ))
        }
    }

    fn create_logical_device(
        &self,
        physical_device: vk::PhysicalDevice,
        mut configuration: DeviceConfiguration,
        queue_infos: &[vk::DeviceQueueCreateInfoBuilder],
    ) -> Result<erupt::DeviceLoader> {
        let device_extensions = self.create_device_extensions(physical_device, &configuration)?;

        #[cfg(feature = "tracing")]
        Self::print_extensions("device", &device_extensions)?;

        let robustness2_name = unsafe { CStr::from_ptr(vk::EXT_ROBUSTNESS_2_EXTENSION_NAME) };
        let robustness2_enabled = device_extensions
            .iter()
            .any(|ext| unsafe { CStr::from_ptr(*ext) == robustness2_name });

        let raytracing_name =
            unsafe { CStr::from_ptr(vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) };
        let raytracing_enabled = device_extensions
            .iter()
            .any(|ext| unsafe { CStr::from_ptr(*ext) == raytracing_name });

        let acceleration_name =
            unsafe { CStr::from_ptr(vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) };
        let acceleration_structure_enabled = device_extensions
            .iter()
            .any(|ext| unsafe { CStr::from_ptr(*ext) == acceleration_name });

        let (
            physical_features,
            physical_features11,
            physical_features12,
            physical_features_synchronization2,
            physical_device_robustness2_features,
            physical_features_raytracing,
            physical_features_acceleration_structure,
        ) = self.collect_physical_device_features(
            physical_device,
            robustness2_enabled,
            raytracing_enabled,
            acceleration_structure_enabled,
        );

        let features = if let Some(features) = configuration.features_v1_0.take() {
            features
        } else {
            vk::PhysicalDeviceFeaturesBuilder::new()
        };
        let features11 = if let Some(features) = configuration.features_v1_1.take() {
            features
        } else {
            vk::PhysicalDeviceVulkan11FeaturesBuilder::new()
        };
        let mut features12 = if let Some(features) = configuration.features_v1_2.take() {
            features
        } else {
            vk::PhysicalDeviceVulkan12FeaturesBuilder::new()
        };
        let device_robustness2_features =
            if let Some(features) = configuration.features_robustness2.take() {
                features
            } else {
                vk::PhysicalDeviceRobustness2FeaturesEXTBuilder::new()
            };
        let features_raytracing = if let Some(features) = configuration.features_raytracing.take() {
            features
        } else {
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHRBuilder::new()
        };
        let features_acceleration_structure =
            if let Some(features) = configuration.features_acceleration_structure.take() {
                features
            } else {
                vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder::new()
            };

        features12 = features12.timeline_semaphore(true);

        #[cfg(feature = "vk-buffer-device-address")]
        {
            features12 = features12.buffer_device_address(true);
        }

        let features_synchronization2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHRBuilder::new().synchronization2(true);

        check_features(
            &physical_features,
            &physical_features11,
            &physical_features12,
            &physical_features_synchronization2,
            &physical_device_robustness2_features,
            &physical_features_raytracing,
            &physical_features_acceleration_structure,
            &features,
            &features11,
            &features12,
            &features_synchronization2,
            &device_robustness2_features,
            &features_raytracing,
            &features_acceleration_structure,
        )?;

        let device_create_info = vk::DeviceCreateInfoBuilder::new()
            .queue_create_infos(queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&features)
            .extend_from(&features11)
            .extend_from(&features12)
            .extend_from(&features_synchronization2);

        let device_create_info = if robustness2_enabled {
            device_create_info.extend_from(&device_robustness2_features)
        } else {
            device_create_info
        };

        let device_create_info = if raytracing_enabled {
            device_create_info.extend_from(&features_raytracing)
        } else {
            device_create_info
        };

        let device_create_info = if acceleration_structure_enabled {
            device_create_info.extend_from(&features_acceleration_structure)
        } else {
            device_create_info
        };

        let device = unsafe {
            erupt::DeviceLoader::new(&self.raw, physical_device, &device_create_info, None)
        }?;

        Ok(device)
    }

    #[cfg(feature = "tracing")]
    fn print_extensions(what: &str, extensions: &[*const i8]) -> Result<()> {
        info!("Loading {} extensions:", what);
        for extension in extensions.iter() {
            let ext = unsafe { CStr::from_ptr(*extension).to_str() }?;
            info!("- {}", ext);
        }
        Ok(())
    }

    fn collect_physical_device_features(
        &self,
        physical_device: vk::PhysicalDevice,
        robustness2_enabled: bool,
        raytracing_enabled: bool,
        acceleration_structure_enabled: bool,
    ) -> (
        vk::PhysicalDeviceFeatures,
        vk::PhysicalDeviceVulkan11FeaturesBuilder,
        vk::PhysicalDeviceVulkan12FeaturesBuilder,
        vk::PhysicalDeviceSynchronization2FeaturesKHRBuilder,
        vk::PhysicalDeviceRobustness2FeaturesEXTBuilder,
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHRBuilder,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder,
    ) {
        let mut physical_features11 = vk::PhysicalDeviceVulkan11FeaturesBuilder::new();
        let mut physical_features12 = vk::PhysicalDeviceVulkan12FeaturesBuilder::new();

        let mut physical_feature_synchronization2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHRBuilder::new();

        let mut physical_device_robustness2_features =
            vk::PhysicalDeviceRobustness2FeaturesEXTBuilder::new();

        let mut physical_features_raytracing =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHRBuilder::new();
        let mut physical_features_acceleration =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder::new();

        let physical_features = unsafe {
            let physical_features = vk::PhysicalDeviceFeatures2Builder::new()
                .extend_from(&mut physical_features11)
                .extend_from(&mut physical_features12)
                .extend_from(&mut physical_feature_synchronization2);

            let physical_features = if robustness2_enabled {
                physical_features.extend_from(&mut physical_device_robustness2_features)
            } else {
                physical_features
            };

            let physical_features = if raytracing_enabled {
                physical_features.extend_from(&mut physical_features_raytracing)
            } else {
                physical_features
            };

            let physical_features = if acceleration_structure_enabled {
                physical_features.extend_from(&mut physical_features_acceleration)
            } else {
                physical_features
            };

            self.raw
                .get_physical_device_features2(physical_device, Some(physical_features.build()))
        };

        (
            physical_features.features,
            physical_features11,
            physical_features12,
            physical_feature_synchronization2,
            physical_device_robustness2_features,
            physical_features_raytracing,
            physical_features_acceleration,
        )
    }

    fn create_device_extensions(
        &self,
        physical_device: vk::PhysicalDevice,
        configuration: &DeviceConfiguration,
    ) -> Result<Vec<*const c_char>> {
        let mut extensions: Vec<*const c_char> = configuration.extensions.clone();

        extensions.push(vk::KHR_SWAPCHAIN_EXTENSION_NAME);
        extensions.push(vk::KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

        // Only keep available extensions.
        let device_extensions = unsafe {
            self.raw
                .enumerate_device_extension_properties(physical_device, None, None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!(
                "Unable to enumerate the device extension properties: {}",
                err
            );
            AscheError::VkResult(err)
        })?;

        // Only keep available extensions.
        Instance::retain_extensions(&device_extensions, &mut extensions);

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
            let surface_supported = unsafe {
                self.raw.get_physical_device_surface_support_khr(
                    physical_device,
                    id.try_into()?,
                    self.surface,
                )
            }
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to get the physical device surface support: {}", err);
                AscheError::VkResult(err)
            })?;

            match target_family {
                vk::QueueFlags::GRAPHICS => {
                    if family.queue_count > 0
                        && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && surface_supported
                    {
                        queue_id = Some(id.try_into()?);
                    }
                }
                vk::QueueFlags::TRANSFER => {
                    if family.queue_count > 0
                        && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && (queue_id.is_none()
                            || !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                && !family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    {
                        queue_id = Some(id.try_into()?);
                    }
                }
                vk::QueueFlags::COMPUTE => {
                    if family.queue_count > 0
                        && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && (queue_id.is_none()
                            || !family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    {
                        queue_id = Some(id.try_into()?);
                    }
                }
                _ => return Err(AscheError::QueueFamilyNotFound("unknown".to_string())),
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
                _ => Err(AscheError::QueueFamilyNotFound("unknown".to_string())),
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

#[allow(unused_mut)]
#[allow(clippy::too_many_arguments)]
fn check_features(
    physical_features: &vk::PhysicalDeviceFeatures,
    physical_features11: &vk::PhysicalDeviceVulkan11FeaturesBuilder,
    physical_features12: &vk::PhysicalDeviceVulkan12FeaturesBuilder,
    physical_features_synchronization2: &vk::PhysicalDeviceSynchronization2FeaturesKHRBuilder,
    physical_device_robustness2_features: &vk::PhysicalDeviceRobustness2FeaturesEXTBuilder,
    physical_features_raytracing: &vk::PhysicalDeviceRayTracingPipelineFeaturesKHRBuilder,
    physical_features_acceleration_structure: &vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder,
    features: &vk::PhysicalDeviceFeaturesBuilder,
    features11: &vk::PhysicalDeviceVulkan11FeaturesBuilder,
    features12: &vk::PhysicalDeviceVulkan12FeaturesBuilder,
    features_synchronization2: &vk::PhysicalDeviceSynchronization2FeaturesKHRBuilder,
    device_robustness2_features: &vk::PhysicalDeviceRobustness2FeaturesEXTBuilder,
    features_raytracing: &vk::PhysicalDeviceRayTracingPipelineFeaturesKHRBuilder,
    features_acceleration_structure: &vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder,
) -> Result<()> {
    let mut physical_features_list: Vec<&str> = Vec::with_capacity(54);
    let mut physical_features11_list: Vec<&str> = Vec::with_capacity(11);
    let mut physical_features12_list: Vec<&str> = Vec::with_capacity(46);
    let mut physical_features_synchronization2_list: Vec<&str> = Vec::with_capacity(1);
    let mut physical_device_robustness2_features_list: Vec<&str> = Vec::with_capacity(3);
    let mut physical_features_raytracing_list: Vec<&str> = Vec::with_capacity(5);
    let mut physical_features_acceleration_structure_list: Vec<&str> = Vec::with_capacity(5);

    let mut features_list: Vec<&str> = Vec::with_capacity(54);
    let mut features11_list: Vec<&str> = Vec::with_capacity(11);
    let mut features12_list: Vec<&str> = Vec::with_capacity(46);
    let mut features_synchronization2_list: Vec<&str> = Vec::with_capacity(1);
    let mut device_robustness2_features_list: Vec<&str> = Vec::with_capacity(3);
    let mut features_raytracing_list: Vec<&str> = Vec::with_capacity(5);
    let mut features_acceleration_structure_list: Vec<&str> = Vec::with_capacity(5);

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

    impl_feature_assemble!(
        physical_features_synchronization2, features_synchronization2, physical_features_synchronization2_list, features_synchronization2_list;
        {
            synchronization2
        }
    );

    impl_feature_assemble!(
        physical_device_robustness2_features, device_robustness2_features, physical_device_robustness2_features_list, device_robustness2_features_list;
        {
            robust_buffer_access2
            robust_image_access2
            null_descriptor
        }
    );

    impl_feature_assemble!(
        physical_features_raytracing, features_raytracing, physical_features_raytracing_list, features_raytracing_list;
        {
            ray_tracing_pipeline
            ray_tracing_pipeline_shader_group_handle_capture_replay
            ray_tracing_pipeline_shader_group_handle_capture_replay_mixed
            ray_tracing_pipeline_trace_rays_indirect
            ray_traversal_primitive_culling
        }
    );

    impl_feature_assemble!(
        physical_features_acceleration_structure, features_acceleration_structure, physical_features_acceleration_structure_list, features_acceleration_structure_list;
        {
            acceleration_structure
            acceleration_structure_capture_replay
            acceleration_structure_host_commands
            acceleration_structure_indirect_build
            descriptor_binding_acceleration_structure_update_after_bind
        }
    );

    let mut missing_features = false;

    #[cfg(feature = "tracing")]
    if !features_list.is_empty() {
        info!("Enabling Vulkan 1.0 features:");
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
    }

    #[cfg(feature = "tracing")]
    if !features11_list.is_empty() {
        info!("Enabling Vulkan 1.1 features:");
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
    }

    #[cfg(feature = "tracing")]
    if !features12_list.is_empty() {
        info!("Enabling Vulkan 1.2 features:");
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
    }

    #[cfg(feature = "tracing")]
    if !features_synchronization2_list.is_empty() {
        info!("Enabling VK_KHR_synchronization2 features:");
        features_synchronization2_list.retain(|&wanted| {
            let found = physical_features_synchronization2_list
                .iter()
                .any(|present| *present == wanted);
            if found {
                info!("- {}", wanted);
                true
            } else {
                #[cfg(feature = "tracing")]
                error!("Unable to find VK_KHR_synchronization2 feature: {}", wanted);
                missing_features = true;
                false
            }
        });
    }

    #[cfg(feature = "tracing")]
    if !device_robustness2_features_list.is_empty() {
        info!("Enabling VK_EXT_robustness2 features:");
        device_robustness2_features_list.retain(|&wanted| {
            let found = physical_device_robustness2_features_list
                .iter()
                .any(|present| *present == wanted);
            if found {
                info!("- {}", wanted);
                true
            } else {
                #[cfg(feature = "tracing")]
                error!("Unable to find VK_EXT_robustness2 feature: {}", wanted);
                missing_features = true;
                false
            }
        });
    }

    #[cfg(feature = "tracing")]
    if !features_raytracing_list.is_empty() {
        info!("Enabling VK_KHR_ray_tracing_pipeline features:");
        features_raytracing_list.retain(|&wanted| {
            let found = physical_features_raytracing_list
                .iter()
                .any(|present| *present == wanted);
            if found {
                info!("- {}", wanted);
                true
            } else {
                #[cfg(feature = "tracing")]
                error!(
                    "Unable to find VK_KHR_ray_tracing_pipeline feature: {}",
                    wanted
                );
                missing_features = true;
                false
            }
        });
    }

    #[cfg(feature = "tracing")]
    if !features_acceleration_structure_list.is_empty() {
        info!("Enabling VK_KHR_acceleration_structure features:");
        features_acceleration_structure_list.retain(|&wanted| {
            let found = physical_features_acceleration_structure_list
                .iter()
                .any(|present| *present == wanted);
            if found {
                info!("- {}", wanted);
                true
            } else {
                #[cfg(feature = "tracing")]
                error!(
                    "Unable to find VK_KHR_acceleration_structure feature: {}",
                    wanted
                );
                missing_features = true;
                false
            }
        });
    }

    if missing_features {
        Err(AscheError::DeviceFeatureMissing)
    } else {
        Ok(())
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            #[cfg(debug_assertions)]
            self.raw
                .destroy_debug_utils_messenger_ext(Some(self.debug_messenger), None);

            self.raw.destroy_surface_khr(Some(self.surface), None);
            self.raw.destroy_instance(None);
        };
    }
}
