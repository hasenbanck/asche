use std::ffi::CString;
use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing1::error;

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
            raw: vk::Queue,
            family_index: u32,
            command_pool_counter: u64,
            context: Arc<Context>,
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

            /// The raw Vulkan queue handle.
            #[inline]
            pub fn raw(&self) -> vk::Queue {
                self.raw
            }

            /// The queue family index of this queue.
            pub fn family_index(&self) -> u32 {
                self.family_index
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
                    .command_buffer(command_buffer.raw())
                    .device_mask(1)];

                let fence = fence.map(|fence| fence.raw());

                let wait_semaphore_infos: Vec<vk::SemaphoreSubmitInfoKHRBuilder> = command_buffer.wait_semaphores.iter().map(|s| s.into()).collect();
                let signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfoKHRBuilder> = command_buffer.signal_semaphores.iter().map(|s| s.into()).collect();

                let submit_info = vk::SubmitInfo2KHRBuilder::new()
                    .command_buffer_infos(&command_buffer_infos)
                    .wait_semaphore_infos(wait_semaphore_infos.as_slice())
                    .signal_semaphore_infos(signal_semaphore_infos.as_slice());

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
                let command_buffer_infos: Vec<vk::CommandBufferSubmitInfoKHRBuilder> = command_buffer.iter().map(|cb| {
                    vk::CommandBufferSubmitInfoKHRBuilder::new()
                        .command_buffer(cb.raw())
                        .device_mask(1)
                })
                .collect();

                let wait_semaphore_infos: Vec<Vec<vk::SemaphoreSubmitInfoKHRBuilder>> = command_buffer.iter().map(|cb| {
                    cb.wait_semaphores.iter().map(|s| s.into()).collect()
                }).collect();

                let signal_semaphore_infos: Vec<Vec<vk::SemaphoreSubmitInfoKHRBuilder>> = command_buffer.iter().map(|cb| {
                    cb.signal_semaphores.iter().map(|s| s.into()).collect()
                }).collect();

                let submit_infos: Vec<vk::SubmitInfo2KHRBuilder> = command_buffer.iter().enumerate().map(|(id, _)| {
                    vk::SubmitInfo2KHRBuilder::new()
                        .command_buffer_infos(&command_buffer_infos[id..id + 1])
                        .wait_semaphore_infos(&wait_semaphore_infos[id])
                        .signal_semaphore_infos(&signal_semaphore_infos[id])
                })
                .collect();

                let fence = fence.map(|fence| fence.raw());

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

            /// Wait for a queue to become idle.
            #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueWaitIdle.html)"]
            pub fn wait_idle(&self) -> Result<()> {
                unsafe { self.context.device.queue_wait_idle(self.raw) }.map_err(|err| {
                    #[cfg(feature = "tracing")]
                    error!("Unable to wait for the queue to become idle: {}", err);
                    AscheError::VkResult(err)
                })
            }

            /// Open a queue debug label region.
            #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueBeginDebugUtilsLabelEXT.html)"]
            pub fn begin_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()> {
                let label = CString::new(label.to_owned())?;
                unsafe {
                    self.context.device.queue_begin_debug_utils_label_ext(
                        self.raw(),
                        &vk::DebugUtilsLabelEXTBuilder::new()
                            .label_name(label.as_c_str())
                            .color(color),
                    )
                }
                Ok(())
            }

            /// Close a queue debug label region.
            #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueEndDebugUtilsLabelEXT.html)"]
            pub fn end_debug_utils_label(&self) {
                unsafe { self.context.device.queue_end_debug_utils_label_ext(self.raw()) };
            }

            /// Insert a label into a queue.
            #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueInsertDebugUtilsLabelEXT.html)"]
            pub fn insert_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()> {
                let label = CString::new(label.to_owned())?;
                unsafe {
                    self.context.device.queue_insert_debug_utils_label_ext(
                        self.raw(),
                        &vk::DebugUtilsLabelEXTBuilder::new()
                            .label_name(label.as_c_str())
                            .color(color),
                    )
                }
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
