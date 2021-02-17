use crate::{AscheError, Result};
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;
use raw_window_handle::RawWindowHandle;
use std::ffi::CStr;
#[cfg(feature = "tracing")]
use tracing::{error, info, level_filters::LevelFilter, warn};

/// Describes how the context should be configured.
pub struct ContextDescriptor<'a> {
    /// Name of the application.
    pub app_name: &'a str,
    /// Version of the application. Use `ash::vk::make_version()` to create the version number.
    pub app_version: u32,
    /// Raw window handle.
    pub handle: &'a raw_window_handle::RawWindowHandle,
}

#[cfg(feature = "tracing")]
struct DebugMessenger {
    ext: ash::extensions::ext::DebugUtils,
    callback: vk::DebugUtilsMessengerEXT,
}

/// Initializes the all Vulkan resources needed to create a device.
pub struct Context {
    _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) surface_loader: ash::extensions::khr::Surface,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) layers: Vec<&'static CStr>,
    #[cfg(feature = "tracing")]
    debug_messenger: Option<DebugMessenger>,
}

impl Context {
    /// Creates a new `Context`.
    pub fn new(descriptor: &ContextDescriptor) -> Result<Self> {
        let entry = ash::Entry::new()?;

        let engine_name = std::ffi::CString::new("asche")?;
        let app_name = std::ffi::CString::new(descriptor.app_name.to_owned())?;

        #[cfg(feature = "tracing")]
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
        let (surface, surface_loader) = Self::create_surface(&entry, &instance, descriptor.handle)?;

        #[cfg(feature = "tracing")]
        {
            let debug_messenger =
                Self::create_debug_messenger(&entry, instance_extensions, &instance);

            Ok(Self {
                _entry: entry,
                instance,
                layers,
                surface,
                surface_loader,
                debug_messenger,
            })
        }
        #[cfg(not(feature = "tracing"))]
        {
            Ok(Self {
                _entry: entry,
                instance: instance,
                layers,
                surface,
                surface_loader,
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
                CStr::from_ptr(props.extension_name.as_ptr())
                    == ash::extensions::ext::DebugUtils::name()
            });

            if debug_utils {
                let ext = ash::extensions::ext::DebugUtils::new(entry, instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                    .message_severity(Self::vulkan_log_level())
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                    .pfn_user_callback(Some(crate::debug_utils_callback));
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
            extensions.push(ash::extensions::ext::DebugUtils::name());
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
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            #[cfg(feature = "tracing")]
            if let Some(messenger) = &self.debug_messenger {
                messenger
                    .ext
                    .destroy_debug_utils_messenger(messenger.callback, None);
            }

            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        };
    }
}
