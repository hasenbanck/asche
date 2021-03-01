use std::sync::Arc;

use erupt::vk;

use crate::{Context, Result};

/// A semaphore that uses the timeline feature.
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
            context: context.clone(),
            raw: semaphore,
        }
    }

    /// Query the timeline value.
    pub fn query_value(&self) -> Result<u64> {
        let value = unsafe {
            self.context
                .device
                .get_semaphore_counter_value(self.raw, None)
                .result()?
        };
        Ok(value)
    }

    /// Sets the timeline value.
    pub fn set_value(&self, timeline_value: u64) -> Result<()> {
        let signal_info = vk::SemaphoreSignalInfoBuilder::new()
            .semaphore(self.raw)
            .value(timeline_value);

        unsafe {
            self.context
                .device
                .signal_semaphore(&signal_info)
                .result()?
        };

        Ok(())
    }

    /// Wait for the given timeline value.
    pub fn wait_for_value(&self, timeline_value: u64) -> Result<()> {
        let semaphores = [self.raw];
        let timeline_values = [timeline_value];
        let wait_info = vk::SemaphoreWaitInfoBuilder::new()
            .semaphores(&semaphores)
            .values(&timeline_values);

        unsafe {
            // 10 sec timeout
            self.context
                .device
                .wait_semaphores(&wait_info, 10000000000)
                .result()?
        };

        Ok(())
    }
}
