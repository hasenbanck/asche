use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing::error;

use crate::command::CommandBuffer;
use crate::{
    AscheError, ComputeCommandBuffer, ComputeCommandPool, Context, GraphicsCommandBuffer,
    GraphicsCommandPool, Result, TransferCommandBuffer, TransferCommandPool,
};

macro_rules! impl_queue {
    (
        #[doc = $doc:expr]
        $queue_name:ident => $pool_name:ident, $buffer_name:ident
    ) => {
        #[doc = $doc]
        pub struct $queue_name {
            pub(crate) inner: Queue,
            /// The queue family index of this queue.
            pub family_index: u32,
        }

        impl $queue_name {
            /// Creates a new queue.
            pub(crate) fn new(
                context: Arc<Context>,
                family_index: u32,
                queue: vk::Queue,
            ) -> Self {
                let raw = Queue::new(context, family_index, queue);
                Self { inner: raw, family_index }
            }

            /// Creates a new command pool. Pools are not cached and are owned by the caller.
            pub fn create_command_pool(&mut self) -> Result<$pool_name> {
                let counter = self.inner.next_command_pool_counter();
                let command_pool = $pool_name::new(
                    self.inner.context.clone(),
                    self.inner.family_index,
                    counter,
                )?;

                Ok(command_pool)
            }

            /// Submits command buffers to a queue.
            ///
            /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueSubmit2KHR.html
            pub fn submit(&self, command_buffer: &$buffer_name) -> Result<()> {
                self.inner.submit(&command_buffer.inner)
            }

            /// TODO Only temporary
            pub fn wait_for_idle(&self) {
                unsafe { self.inner.context.device.queue_wait_idle(self.inner.raw).unwrap() };
            }
        }
    };
}

impl_queue!(
    #[doc = "A queue for compute operations."]
    ComputeQueue => ComputeCommandPool, ComputeCommandBuffer
);

impl_queue!(
    #[doc = "A queue for graphics operations."]
    GraphicsQueue => GraphicsCommandPool, GraphicsCommandBuffer
);

impl_queue!(
    #[doc = "A queue for transfer operations."]
    TransferQueue => TransferCommandPool, TransferCommandBuffer
);

/// The inner queue that abstracts common functionality.
pub(crate) struct Queue {
    pub(crate) raw: vk::Queue,
    family_index: u32,
    command_pool_counter: u64,
    context: Arc<Context>,
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe {
            self.context.device.queue_wait_idle(self.raw).unwrap();
        };
    }
}

impl Queue {
    fn new(context: Arc<Context>, family_index: u32, queue: vk::Queue) -> Self {
        Self {
            raw: queue,
            family_index,
            command_pool_counter: 0,
            context,
        }
    }

    #[inline]
    fn next_command_pool_counter(&mut self) -> u64 {
        let counter = self.command_pool_counter;
        self.command_pool_counter += 1;
        counter
    }

    #[inline]
    fn submit(&self, command_buffer: &CommandBuffer) -> Result<()> {
        let command_buffer_infos = [vk::CommandBufferSubmitInfoKHRBuilder::new()
            .command_buffer(command_buffer.buffer)
            .device_mask(1)];

        let wait_semaphore_infos = [vk::SemaphoreSubmitInfoKHRBuilder::new()
            .semaphore(command_buffer.timeline_semaphore)
            .value(command_buffer.wait_value)
            .stage_mask(vk::PipelineStageFlags2KHR::NONE_KHR)
            .device_index(1)];

        let signal_semaphore_infos = [vk::SemaphoreSubmitInfoKHRBuilder::new()
            .semaphore(command_buffer.timeline_semaphore)
            .value(command_buffer.signal_value)
            .stage_mask(vk::PipelineStageFlags2KHR::NONE_KHR)
            .device_index(1)];

        let submit_info = vk::SubmitInfo2KHRBuilder::new()
            .command_buffer_infos(&command_buffer_infos)
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos);

        unsafe {
            self.context
                .device
                .queue_submit2_khr(self.raw, &[submit_info], None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to queue and submit a command buffer: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(())
    }
}
