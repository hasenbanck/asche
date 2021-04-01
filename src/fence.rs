use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing::error;

use crate::context::Context;
use crate::{AscheError, Result};

/// A fence.
#[derive(Debug)]
pub struct Fence {
    pub(crate) raw: vk::Fence,
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

    /// Wait for the fence.
    pub fn wait(&self) -> Result<()> {
        unsafe {
            self.context
                .device
                .wait_for_fences(&[self.raw], true, u64::MAX)
                .map_err(|err| {
                    #[cfg(feature = "tracing")]
                    error!("Unable to wait for a fence: {}", err);
                    AscheError::VkResult(err)
                })
        }
    }

    /// Resets the fence.
    pub fn reset(&self) -> Result<()> {
        unsafe {
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
}
