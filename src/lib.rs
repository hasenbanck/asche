#![warn(missing_docs)]
//! Provides an abstraction layer above ash to easier use Vulkan in Rust with minimal dependencies.

use std::rc::Rc;

use ash::version::DeviceV1_0;
#[cfg(feature = "tracing")]
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, error, info, warn};

pub use {
    device::{Device, DeviceDescriptor, QueuePriorityDescriptor},
    error::AscheError,
    instance::{Instance, InstanceDescriptor},
};

pub(crate) mod device;
pub(crate) mod error;
pub(crate) mod instance;
pub(crate) mod swapchain;

pub(crate) type Result<T> = std::result::Result<T, AscheError>;

/// Construct a `*const std::os::raw::c_char` from a string
#[macro_export]
macro_rules! cstr {
    ($s:expr) => {
        concat!($s, "\0") as *const str as *const c_char
    };
}

/// Callback function for the debug utils logging.
#[cfg(feature = "tracing")]
pub(crate) unsafe extern "system" fn debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if std::thread::panicking() {
        return vk::FALSE;
    }

    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let ty = format!("{:?}", message_type);

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!("{} - {:?}", ty, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!("{} - {:?}", ty, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!("{} - {:?}", ty, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            debug!("{} - {:?}", ty, message)
        }
        _ => {
            warn!("{} - {:?}", ty, message);
        }
    }

    vk::FALSE
}

/// A context used internally.
pub(crate) struct Context {
    pub(crate) instance: Instance,
    pub(crate) logical_device: ash::Device,
    pub(crate) physical_device: vk::PhysicalDevice,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { self.logical_device.destroy_device(None) };
    }
}

/// Wraps a render pass.
pub struct RenderPass {
    pub(crate) context: Rc<Context>,
    /// The raw vk::RenderPass
    pub raw: vk::RenderPass,
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_render_pass(self.raw, None);
        };
    }
}

/// Wraps a pipeline layout.
pub struct PipelineLayout {
    pub(crate) context: Rc<Context>,
    /// The raw vk::PipelineLayout
    pub raw: vk::PipelineLayout,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_pipeline_layout(self.raw, None);
        };
    }
}

/// Wraps a pipeline.
pub struct Pipeline {
    pub(crate) context: Rc<Context>,
    /// The raw ck::Pipeline.
    pub raw: vk::Pipeline,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.logical_device.destroy_pipeline(self.raw, None);
        };
    }
}

/// Wraps a shader module.
pub struct ShaderModule {
    pub(crate) context: Rc<Context>,
    /// The raw vk::ShaderModule.
    pub raw: vk::ShaderModule,
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_shader_module(self.raw, None);
        };
    }
}

/// A wrapped command pool.
pub struct CommandPool {
    pub(crate) context: Rc<Context>,
    pub(crate) raw: vk::CommandPool,
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_command_pool(self.raw, None);
        };
    }
}

/// A wrapped command buffer.
pub struct CommandBuffer {
    pub(crate) context: Rc<Context>,
    pub(crate) raw: vk::CommandBuffer,
}

impl CommandBuffer {
    /// Sets the viewport.
    pub fn set_viewport(&self, viewport: vk::Viewport) {
        unsafe {
            self.context
                .logical_device
                .cmd_set_viewport(self.raw, 0, &[viewport]);
        };
    }

    /// Sets the scissor rectangle.
    pub fn set_scissor(&self, scissor_rect: vk::Rect2D) {
        unsafe {
            self.context
                .logical_device
                .cmd_set_scissor(self.raw, 0, &[scissor_rect]);
        };
    }
}
