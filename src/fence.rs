use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing1::error;

use crate::context::Context;
use crate::{AscheError, Result};

/// A fence.
#[derive(Debug)]
pub struct Fence {
    raw: vk::Fence,
    context: Arc<Context>,
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_fence(Some(self.raw), None);
        };
    }
}

impl Fence {
    /// Creates a new fence.
    pub(crate) fn new(context: Arc<Context>, fence: vk::Fence) -> Self {
        Self {
            context,
            raw: fence,
        }
    }

    /// The raw Vulkan fence handle.
    #[inline]
    pub fn raw(&self) -> vk::Fence {
        self.raw
    }

    /// Wait for the fence.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn wait(&self) -> Result<()> {
        self.context
            .device
            .wait_for_fences(&[self.raw], true, u64::MAX)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to wait for a fence: {}", err);
                AscheError::VkResult(err)
            })
    }

    /// Resets the fence.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn reset(&self) -> Result<()> {
        self.context
            .device
            .reset_fences(&[self.raw])
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to reset a fence: {}", err);
                AscheError::VkResult(err)
            })
    }
}
