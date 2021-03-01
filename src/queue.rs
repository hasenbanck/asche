use std::sync::Arc;

use erupt::{vk, ExtendableFrom};

use crate::command::CommandBuffer;
use crate::{
    ComputeCommandBuffer, ComputeCommandPool, Context, GraphicsCommandBuffer, GraphicsCommandPool,
    Result, TransferCommandBuffer, TransferCommandPool,
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

            /// Executes a command buffer on a queue.
            pub fn execute(&self, command_buffer: &$buffer_name) -> Result<()> {
                self.inner.execute(&command_buffer.inner)
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
    fn execute(&self, command_buffer: &CommandBuffer) -> Result<()> {
        let semaphores = [command_buffer.timeline_semaphore];
        let timeline_wait_values = [command_buffer.timeline_wait_value];
        let timeline_signal_values = [command_buffer.timeline_signal_value];
        let command_buffers = [command_buffer.buffer];
        let wait_dst_stage_mask = [vk::PipelineStageFlags::TOP_OF_PIPE];

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfoBuilder::new()
            .wait_semaphore_values(&timeline_wait_values)
            .signal_semaphore_values(&timeline_signal_values);

        let submit_info = vk::SubmitInfoBuilder::new()
            .wait_dst_stage_mask(&wait_dst_stage_mask)
            .command_buffers(&command_buffers)
            .wait_semaphores(&semaphores)
            .signal_semaphores(&semaphores)
            .extend_from(&mut timeline_info);

        unsafe {
            self.context
                .device
                .queue_submit(self.raw, &[submit_info], None)
                .result()?
        };

        Ok(())
    }
}
