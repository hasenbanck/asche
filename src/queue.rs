use std::sync::Arc;

use ash::version::{DeviceV1_0, DeviceV1_2};
use ash::vk;
use ash::vk::Handle;

use crate::command::CommandBuffer;
use crate::{
    ComputeCommandBuffer, ComputeCommandPool, Context, GraphicsCommandBuffer, GraphicsCommandPool,
    QueueType, Result, TransferCommandBuffer, TransferCommandPool,
};

macro_rules! impl_queue {
    (
        #[doc = $doc:expr]
        $queue_name:ident => $pool_name:ident, $buffer_name:ident
    ) => {
        #[doc = $doc]
        pub struct $queue_name {
            pub(crate) inner: Queue,
        }

        impl $queue_name {
            /// Creates a new queue.
            pub(crate) fn new(
                context: Arc<Context>,
                family_index: u32,
                queue: vk::Queue,
            ) -> Result<Self> {
                let queue = Queue::new(context, QueueType::Compute, family_index, queue)?;

                Ok(Self { inner: queue })
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

            /// Query the timeline value.
            pub fn query_timeline_value(&self) -> Result<u64> {
                self.inner.query_timeline_value()
            }

            /// Sets the timeline value.
            pub fn set_timeline_value(&self, timeline_value: u64) -> Result<()> {
                self.inner.set_timeline_value(timeline_value)
            }

            /// Wait for the given timeline value.
            pub fn wait_for_timeline_value(&self, timeline_value: u64) -> Result<()> {
                self.inner.wait_for_timeline_value(timeline_value)
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
    context: Arc<Context>,
    family_index: u32,
    pub(crate) raw: vk::Queue,
    timeline: vk::Semaphore,
    command_pool_counter: u64,
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .queue_wait_idle(self.raw)
                .unwrap();
            self.context
                .logical_device
                .destroy_semaphore(self.timeline, None);
        };
    }
}

impl Queue {
    fn new(
        context: Arc<Context>,
        queue_type: QueueType,
        family_index: u32,
        queue: vk::Queue,
    ) -> Result<Self> {
        let mut create_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_info = vk::SemaphoreCreateInfo::builder().push_next(&mut create_info);
        let timeline = unsafe {
            context
                .logical_device
                .create_semaphore(&semaphore_info, None)?
        };

        context.set_object_name(
            &format!("Queue {}", queue_type),
            vk::ObjectType::QUEUE,
            queue.as_raw(),
        )?;

        context.set_object_name(
            &format!("Semaphore Timeline {}", queue_type),
            vk::ObjectType::SEMAPHORE,
            timeline.as_raw(),
        )?;

        Ok(Self {
            context,
            family_index,
            raw: queue,
            timeline,
            command_pool_counter: 0,
        })
    }

    #[inline]
    fn next_command_pool_counter(&mut self) -> u64 {
        let counter = self.command_pool_counter;
        self.command_pool_counter += 1;
        counter
    }

    #[inline]
    fn execute(&self, command_buffer: &CommandBuffer) -> Result<()> {
        let semaphores = [self.timeline];
        let timeline_wait_values = [command_buffer.timeline_wait_value];
        let timeline_signal_values = [command_buffer.timeline_signal_value];
        let command_buffers = [command_buffer.buffer];

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(&timeline_wait_values)
            .signal_semaphore_values(&timeline_signal_values);

        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&[ash::vk::PipelineStageFlags::ALL_COMMANDS])
            .command_buffers(&command_buffers)
            .wait_semaphores(&semaphores)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline_info);

        unsafe {
            self.context.logical_device.queue_submit(
                self.raw,
                &[submit_info.build()],
                vk::Fence::null(),
            )?
        };

        Ok(())
    }

    #[inline]
    fn query_timeline_value(&self) -> Result<u64> {
        let value = unsafe {
            self.context
                .logical_device
                .get_semaphore_counter_value(self.timeline)?
        };
        Ok(value)
    }

    #[inline]
    fn set_timeline_value(&self, timeline_value: u64) -> Result<()> {
        let signal_info = vk::SemaphoreSignalInfo::builder()
            .semaphore(self.timeline)
            .value(timeline_value);

        unsafe { self.context.logical_device.signal_semaphore(&signal_info)? };

        Ok(())
    }

    #[inline]
    fn wait_for_timeline_value(&self, timeline_value: u64) -> Result<()> {
        let semaphores = [self.timeline];
        let timeline_values = [timeline_value];
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(&semaphores)
            .values(&timeline_values);

        unsafe {
            self.context
                .logical_device
                .wait_semaphores(&wait_info, 5000000000)? // 5 sec timeout
        };

        Ok(())
    }
}
