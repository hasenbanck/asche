use std::sync::Arc;

use ash::version::{DeviceV1_0, DeviceV1_2};
use ash::vk;

use crate::{Context, Result};

/// A semaphore that uses the timeline feature.
pub struct TimelineSemaphore {
    context: Arc<Context>,
    pub(crate) raw: vk::Semaphore,
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_semaphore(self.raw, None);
        };
    }
}

impl TimelineSemaphore {
    /// Creates a new timeline semaphore.
    pub(crate) fn new(context: Arc<Context>, semaphore: vk::Semaphore) -> Self {
        Self {
            context: context.clone(),
            raw: semaphore,
        }
    }

    /// Query the timeline value.
    pub fn query_value(&self) -> Result<u64> {
        let value = unsafe {
            self.context
                .logical_device
                .get_semaphore_counter_value(self.raw)?
        };
        Ok(value)
    }

    /// Sets the timeline value.
    pub fn set_value(&self, timeline_value: u64) -> Result<()> {
        let signal_info = vk::SemaphoreSignalInfo::builder()
            .semaphore(self.raw)
            .value(timeline_value);

        unsafe { self.context.logical_device.signal_semaphore(&signal_info)? };

        Ok(())
    }

    /// Wait for the given timeline value.
    pub fn wait_for_value(&self, timeline_value: u64) -> Result<()> {
        let semaphores = [self.raw];
        let timeline_values = [timeline_value];
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(&semaphores)
            .values(&timeline_values);

        unsafe {
            self.context
                .logical_device
                .wait_semaphores(&wait_info, 10000000000)? // 10 sec timeout
        };

        Ok(())
    }
}
