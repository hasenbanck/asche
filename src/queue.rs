use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "smallvec")]
use smallvec::SmallVec;
#[cfg(feature = "tracing")]
use tracing::error;

use crate::context::Context;
use crate::{
    AscheError, ComputeCommandBuffer, ComputeCommandPool, GraphicsCommandBuffer,
    GraphicsCommandPool, Result, TransferCommandBuffer, TransferCommandPool,
};

macro_rules! impl_queue {
    (
        #[doc = $doc:expr]
        $queue_name:ident => $pool_name:ident, $buffer_name:ident
    ) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $queue_name {
            /// The queue family index of this queue.
            pub family_index: u32,
            pub(crate) raw: vk::Queue,
            command_pool_counter: u64,
            context: Arc<Context>,
        }

        impl Drop for $queue_name {
            fn drop(&mut self) {
                unsafe {
                    self.context.device.queue_wait_idle(self.raw).unwrap();
                };
            }
        }

        impl $queue_name {
            pub(crate) fn new(context: Arc<Context>, family_index: u32, queue: vk::Queue) -> Self {
                Self {
                    raw: queue,
                    family_index,
                    command_pool_counter: 0,
                    context,
                }
            }

            /// Creates a new command pool. Pools are not cached and are owned by the caller.
            pub fn create_command_pool(&mut self) -> Result<$pool_name> {
                let counter = self.command_pool_counter;
                let command_pool =
                    $pool_name::new(self.context.clone(), self.family_index, counter)?;

                self.command_pool_counter += 1;

                Ok(command_pool)
            }

            /// Submits a command buffer to a queue.
            #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueSubmit2KHR.html)"]
            pub fn submit(&mut self, command_buffer: &$buffer_name) -> Result<()> {
                let command_buffer_infos = [vk::CommandBufferSubmitInfoKHRBuilder::new()
                    .command_buffer(command_buffer.raw)
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

                let submit_info = [vk::SubmitInfo2KHRBuilder::new()
                    .command_buffer_infos(&command_buffer_infos)
                    .wait_semaphore_infos(&wait_semaphore_infos)
                    .signal_semaphore_infos(&signal_semaphore_infos)];

                unsafe {
                    self.context
                        .device
                        .queue_submit2_khr(self.raw, &submit_info, None)
                }
                .map_err(|err| {
                    #[cfg(feature = "tracing")]
                    error!("Unable to queue and submit a command buffer: {}", err);
                    AscheError::VkResult(err)
                })?;

                Ok(())
            }

            /// Submit command buffers to a queue.
            #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueSubmit2KHR.html)"]
            pub fn submit_all(&mut self, command_buffer: &[$buffer_name]) -> Result<()> {
                let command_buffer_infos = command_buffer.iter().map(|cb| {
                    vk::CommandBufferSubmitInfoKHRBuilder::new()
                        .command_buffer(cb.raw)
                        .device_mask(1)
                });

                #[cfg(feature = "smallvec")]
                let command_buffer_infos = command_buffer_infos
                    .collect::<SmallVec<[vk::CommandBufferSubmitInfoKHRBuilder; 4]>>();

                #[cfg(not(feature = "smallvec"))]
                let command_buffer_infos =
                    command_buffer_infos.collect::<Vec<vk::CommandBufferSubmitInfoKHRBuilder>>();

                let wait_semaphore_infos = command_buffer.iter().map(|cb| {
                    vk::SemaphoreSubmitInfoKHRBuilder::new()
                        .semaphore(cb.timeline_semaphore)
                        .value(cb.wait_value)
                        .stage_mask(vk::PipelineStageFlags2KHR::NONE_KHR)
                        .device_index(1)
                });

                #[cfg(feature = "smallvec")]
                let wait_semaphore_infos = wait_semaphore_infos
                    .collect::<SmallVec<[vk::SemaphoreSubmitInfoKHRBuilder; 4]>>();

                #[cfg(not(feature = "smallvec"))]
                let wait_semaphore_infos =
                    wait_semaphore_infos.collect::<Vec<vk::SemaphoreSubmitInfoKHRBuilder>>();

                let signal_semaphore_infos = command_buffer.iter().map(|cb| {
                    vk::SemaphoreSubmitInfoKHRBuilder::new()
                        .semaphore(cb.timeline_semaphore)
                        .value(cb.signal_value)
                        .stage_mask(vk::PipelineStageFlags2KHR::NONE_KHR)
                        .device_index(1)
                });

                #[cfg(feature = "smallvec")]
                let signal_semaphore_infos = signal_semaphore_infos
                    .collect::<SmallVec<[vk::SemaphoreSubmitInfoKHRBuilder; 4]>>();

                #[cfg(not(feature = "smallvec"))]
                let signal_semaphore_infos =
                    signal_semaphore_infos.collect::<Vec<vk::SemaphoreSubmitInfoKHRBuilder>>();

                let submit_infos = command_buffer.iter().enumerate().map(|(id, _)| {
                    vk::SubmitInfo2KHRBuilder::new()
                        .command_buffer_infos(&command_buffer_infos[id..id + 1])
                        .wait_semaphore_infos(&wait_semaphore_infos[id..id + 1])
                        .signal_semaphore_infos(&signal_semaphore_infos[id..id + 1])
                });

                #[cfg(feature = "smallvec")]
                let submit_infos =
                    submit_infos.collect::<SmallVec<[vk::SubmitInfo2KHRBuilder; 4]>>();

                #[cfg(not(feature = "smallvec"))]
                let submit_infos = submit_infos.collect::<Vec<vk::SubmitInfo2KHRBuilder>>();

                unsafe {
                    self.context
                        .device
                        .queue_submit2_khr(self.raw, &submit_infos, None)
                }
                .map_err(|err| {
                    #[cfg(feature = "tracing")]
                    error!("Unable to queue and submit command buffers: {}", err);
                    AscheError::VkResult(err)
                })?;

                Ok(())
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
