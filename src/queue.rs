use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "smallvec")]
use smallvec::SmallVec;
#[cfg(feature = "tracing")]
use tracing::error;

use crate::context::Context;
use crate::{
    AscheError, ComputeCommandBuffer, ComputeCommandPool, Fence, GraphicsCommandBuffer,
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
            pub fn submit(&mut self, command_buffer: &$buffer_name, fence: Option<&Fence>) -> Result<()> {
                let command_buffer_infos = [vk::CommandBufferSubmitInfoKHRBuilder::new()
                    .command_buffer(command_buffer.raw)
                    .device_mask(1)];

                let fence = if let Some(fence) = fence {
                    Some(fence.raw)
                } else {
                    None
                };

                #[cfg(feature = "smallvec")]
                let wait_semaphore_infos: SmallVec<[vk::SemaphoreSubmitInfoKHRBuilder; 1]> =
                if let Some(wait_semaphore) = command_buffer.wait_semaphore.as_ref() {
                    SmallVec::from_buf([wait_semaphore.into()])
                } else {
                    SmallVec::new()
                };

                #[cfg(not(feature = "smallvec"))]
                let wait_semaphore_infos: Vec<vk::SemaphoreSubmitInfoKHRBuilder> =
                if let Some(wait_semaphore) = command_buffer.wait_semaphore.as_ref() {
                    vec![wait_semaphore.into()]
                } else {
                    vec![]
                };

                #[cfg(feature = "smallvec")]
                let signal_semaphore_infos: SmallVec<[vk::SemaphoreSubmitInfoKHRBuilder; 1]> =
                if let Some(signal_semaphore) = command_buffer.signal_semaphore.as_ref() {
                    SmallVec::from_buf([signal_semaphore.into()])
                } else {
                    SmallVec::new()
                };

                #[cfg(not(feature = "smallvec"))]
                let signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfoKHRBuilder> =
                if let Some(signal_semaphore) = command_buffer.signal_semaphore.as_ref() {
                    vec![signal_semaphore.into()]
                } else {
                    vec![]
                };

                let mut submit_info = vk::SubmitInfo2KHRBuilder::new().command_buffer_infos(&command_buffer_infos);

                submit_info = if signal_semaphore_infos.len() > 0 {
                    submit_info.signal_semaphore_infos(signal_semaphore_infos.as_slice())
                } else {
                    submit_info
                };

                submit_info = if wait_semaphore_infos.len() > 0 {
                    submit_info.wait_semaphore_infos(wait_semaphore_infos.as_slice())
                } else {
                    submit_info
                };

                unsafe {
                    self.context
                        .device
                        .queue_submit2_khr(self.raw, &[submit_info], fence)
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
            pub fn submit_all(&mut self, command_buffer: &[$buffer_name], fence: Option<&Fence>) -> Result<()> {
                let command_buffer_infos = command_buffer.iter().map(|cb| {
                    vk::CommandBufferSubmitInfoKHRBuilder::new()
                        .command_buffer(cb.raw)
                        .device_mask(1)
                });

                #[cfg(feature = "smallvec")]
                let command_buffer_infos = command_buffer_infos.collect::<SmallVec<[vk::CommandBufferSubmitInfoKHRBuilder; 4]>>();

                #[cfg(not(feature = "smallvec"))]
                let command_buffer_infos = command_buffer_infos.collect::<Vec<vk::CommandBufferSubmitInfoKHRBuilder>>();

                #[cfg(feature = "smallvec")]
                let mut wait_semaphore_infos: SmallVec<[vk::SemaphoreSubmitInfoKHRBuilder; 4]> = SmallVec::new();

                #[cfg(not(feature = "smallvec"))]
                let mut wait_semaphore_infos: Vec<vk::SemaphoreSubmitInfoKHRBuilder> = Vec::with_capacity(4);

                #[cfg(feature = "smallvec")]
                let mut signal_semaphore_infos: SmallVec<[vk::SemaphoreSubmitInfoKHRBuilder; 4]> = SmallVec::new();

                #[cfg(not(feature = "smallvec"))]
                let mut signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfoKHRBuilder> = Vec::with_capacity(4);

                 for cb in command_buffer.iter() {
                    if let Some(wait_semaphore) = cb.wait_semaphore.as_ref() {
                        let wait_semaphore: vk::SemaphoreSubmitInfoKHRBuilder = wait_semaphore.into();
                        wait_semaphore_infos.push(wait_semaphore);
                    } else {
                        return Err(AscheError::MissingWaitSemaphore);
                    };
                    if let Some(signal_semaphore) = cb.signal_semaphore.as_ref() {
                        let signal_semaphore: vk::SemaphoreSubmitInfoKHRBuilder = signal_semaphore.into();
                        signal_semaphore_infos.push(signal_semaphore);
                    } else {
                        return Err(AscheError::MissingSignalSemaphore);
                    };
                };

                let submit_infos = command_buffer.iter().enumerate().map(|(id, _)| {
                    vk::SubmitInfo2KHRBuilder::new()
                        .command_buffer_infos(&command_buffer_infos[id..id + 1])
                        .wait_semaphore_infos(&wait_semaphore_infos[id..id + 1])
                        .signal_semaphore_infos(&signal_semaphore_infos[id..id + 1])
                });

                #[cfg(feature = "smallvec")]
                let submit_infos = submit_infos.collect::<SmallVec<[vk::SubmitInfo2KHRBuilder; 4]>>();

                #[cfg(not(feature = "smallvec"))]
                let submit_infos = submit_infos.collect::<Vec<vk::SubmitInfo2KHRBuilder>>();

                let fence = if let Some(fence) = fence {
                    Some(fence.raw)
                } else {
                    None
                };

                unsafe {
                    self.context
                        .device
                        .queue_submit2_khr(self.raw, &submit_infos, fence)
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
