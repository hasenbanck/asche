#![warn(missing_docs)]
//! Provides an abstraction layer above ash to easier use Vulkan in Rust with minimal dependencies.

use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, error, info, warn};

pub use {
    command::{CommandBuffer, CommandPool},
    device::{Device, DeviceDescriptor, QueuePriorityDescriptor},
    error::AscheError,
    instance::{Instance, InstanceDescriptor},
};

use crate::context::Context;

pub(crate) mod command;
pub(crate) mod context;
pub(crate) mod device;
pub(crate) mod error;
pub(crate) mod instance;
pub(crate) mod swapchain;

pub(crate) type Result<T> = std::result::Result<T, AscheError>;

/// Callback function for the debug utils logging.
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

    #[cfg(feature = "tracing")]
    {
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
    }

    #[cfg(not(feature = "tracing"))]
    {
        match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
                println!("ERROR: {} - {:?}", ty, message)
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
                println!("WARN: {} - {:?}", ty, message)
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
                println!("INFO: {} - {:?}", ty, message)
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
                println!("DEBUG: {} - {:?}", ty, message)
            }
            _ => {
                println!("WARN: {} - {:?}", ty, message);
            }
        }
    }

    vk::FALSE
}

/// Type of a queue.
#[derive(Copy, Clone)]
pub enum QueueType {
    /// Compute queue.
    Compute = 0,
    /// Graphics queue.
    Graphics,
    /// Transfer queue.
    Transfer,
}

/// Defines which queue to wait for.
#[derive(Copy, Clone)]
pub enum WaitForQueueType {
    /// Wait for all queues to finish.
    All = 0,
    /// Compute queue.
    Compute,
    /// Graphics queue.
    Graphics,
    /// Transfer queue.
    Transfer,
}

/// Abstracts a Vulkan queue.
pub struct Queue {
    pub(crate) family_index: u32,
    pub(crate) raw: vk::Queue,
}

/// Wraps a render pass.
pub struct RenderPass {
    pub(crate) context: Arc<Context>,
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
    pub(crate) context: Arc<Context>,
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
    pub(crate) context: Arc<Context>,
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
    pub(crate) context: Arc<Context>,
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
