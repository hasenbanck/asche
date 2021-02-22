use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;

use crate::context::Context;

/// Wraps a Vulkan queue.
pub struct Queue {
    pub(crate) context: Arc<Context>,
    pub(crate) family_index: u32,
    pub(crate) raw: vk::Queue,
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .queue_wait_idle(self.raw)
                .unwrap()
        };
    }
}
