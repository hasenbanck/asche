use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing1::error;

use crate::context::Context;
use crate::{AscheError, Result};

/// A handle of a binary semaphore.
#[derive(Debug, Clone, Copy)]
pub struct BinarySemaphoreHandle(pub(crate) vk::Semaphore);

/// A handle of a timeline semaphore.
#[derive(Debug, Clone, Copy)]
pub struct TimelineSemaphoreHandle(pub(crate) vk::Semaphore);

/// A binary semaphore.
#[derive(Debug, Clone)]
pub struct BinarySemaphore {
    raw: vk::Semaphore,
    context: Arc<Context>,
}

impl Drop for BinarySemaphore {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_semaphore(Some(self.raw), None);
        };
    }
}

impl BinarySemaphore {
    /// Creates a new semaphore.
    pub(crate) fn new(context: Arc<Context>, semaphore: vk::Semaphore) -> Self {
        Self {
            context,
            raw: semaphore,
        }
    }

    /// Returns the handle of the binary semaphore.
    pub fn handle(&self) -> BinarySemaphoreHandle {
        BinarySemaphoreHandle(self.raw)
    }

    /// The raw Vulkan semaphore handle.
    #[inline]
    pub(crate) fn raw(&self) -> vk::Semaphore {
        self.raw
    }
}

/// A semaphore that uses the timeline feature.
#[derive(Debug)]
pub struct TimelineSemaphore {
    pub(crate) raw: vk::Semaphore,
    context: Arc<Context>,
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_semaphore(Some(self.raw), None);
        };
    }
}

impl TimelineSemaphore {
    /// Creates a new timeline semaphore.
    pub(crate) fn new(context: Arc<Context>, semaphore: vk::Semaphore) -> Self {
        Self {
            context,
            raw: semaphore,
        }
    }

    /// Returns the handle of the timeline semaphore.
    pub fn handle(&self) -> TimelineSemaphoreHandle {
        TimelineSemaphoreHandle(self.raw)
    }

    /// Query the timeline value.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn query_value(&self) -> Result<u64> {
        let value = self
            .context
            .device
            .get_semaphore_counter_value(self.raw)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to get a semaphore counter value: {}", err);
                AscheError::VkResult(err)
            })?;
        Ok(value)
    }

    /// Sets the timeline value.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn set_value(&self, timeline_value: u64) -> Result<()> {
        let signal_info = vk::SemaphoreSignalInfoBuilder::new()
            .semaphore(self.raw)
            .value(timeline_value);

        self.context
            .device
            .signal_semaphore(&signal_info)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to signal a semaphore: {}", err);
                AscheError::VkResult(err)
            })?;

        Ok(())
    }

    /// Wait for the given timeline value.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn wait_for_value(&self, timeline_value: u64) -> Result<()> {
        let semaphores = [self.raw];
        let timeline_values = [timeline_value];
        let wait_info = vk::SemaphoreWaitInfoBuilder::new()
            .semaphores(&semaphores)
            .values(&timeline_values);

        // 10 sec timeout
        self.context
            .device
            .wait_semaphores(&wait_info, 10000000000)
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to wait for a semaphore: {}", err);
                AscheError::VkResult(err)
            })?;

        Ok(())
    }
}
