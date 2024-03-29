//! Implements command pools and command buffers.

use std::convert::TryInto;
use std::ffi::{c_void, CString};
use std::sync::Arc;

use erupt::vk;
use erupt::vk::ClearValue;
#[cfg(feature = "tracing")]
use tracing1::error;

use crate::context::Context;
use crate::{
    AscheError, BinarySemaphoreHandle, ComputePipeline, GraphicsPipeline, QueueType,
    RayTracingPipeline, RenderPass, RenderPassColorAttachmentDescriptor,
    RenderPassDepthAttachmentDescriptor, Result, Swapchain, TimelineSemaphoreHandle,
};

/// Defines the semaphore a command buffer will use on wait and signal.
#[derive(Debug)]
pub enum CommandBufferSemaphore {
    /// A binary semaphore.
    Binary {
        /// Handle of the binary semaphore.
        semaphore: BinarySemaphoreHandle,
        /// The stage for this semaphore.
        stage: vk::PipelineStageFlags2KHR,
    },
    /// A timeline semaphore.
    Timeline {
        /// Handle of the timeline semaphore.
        semaphore: TimelineSemaphoreHandle,
        /// The stage for this semaphore.
        stage: vk::PipelineStageFlags2KHR,
        /// The timeline value to wait / signal.
        value: u64,
    },
}

/// Defines the internal representation of the semaphore a command buffer will use on wait and signal.
#[derive(Debug)]
pub(crate) enum CommandBufferSemaphoreInner {
    Binary {
        semaphore: vk::Semaphore,
        stage: vk::PipelineStageFlags2KHR,
    },
    Timeline {
        semaphore: vk::Semaphore,
        stage: vk::PipelineStageFlags2KHR,
        value: u64,
    },
}

impl From<&CommandBufferSemaphore> for CommandBufferSemaphoreInner {
    fn from(s: &CommandBufferSemaphore) -> Self {
        match s {
            CommandBufferSemaphore::Binary { semaphore, stage } => {
                CommandBufferSemaphoreInner::Binary {
                    semaphore: semaphore.0,
                    stage: *stage,
                }
            }
            CommandBufferSemaphore::Timeline {
                semaphore,
                stage,
                value,
            } => CommandBufferSemaphoreInner::Timeline {
                semaphore: semaphore.0,
                stage: *stage,
                value: *value,
            },
        }
    }
}

impl From<&CommandBufferSemaphoreInner> for vk::SemaphoreSubmitInfoKHRBuilder<'_> {
    fn from(s: &CommandBufferSemaphoreInner) -> Self {
        match s {
            CommandBufferSemaphoreInner::Binary { semaphore, stage } => {
                vk::SemaphoreSubmitInfoKHRBuilder::new()
                    .semaphore(*semaphore)
                    .stage_mask(*stage)
                    .device_index(1)
            }
            CommandBufferSemaphoreInner::Timeline {
                semaphore,
                value,
                stage,
            } => vk::SemaphoreSubmitInfoKHRBuilder::new()
                .semaphore(*semaphore)
                .value(*value)
                .stage_mask(*stage)
                .device_index(1),
        }
    }
}

macro_rules! impl_command_pool {
    (
        #[doc = $doc:expr]
        $pool_name:ident => $buffer_name:ident, $queue_type:expr
    ) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $pool_name {
            raw: vk::CommandPool,
            queue_type: QueueType,
            command_buffer_counter: u64,
            context: Arc<Context>,
        }

        impl Drop for $pool_name {
            fn drop(&mut self) {
                unsafe {
                    self.context
                        .device
                        .destroy_command_pool(Some(self.raw), None);
                };
            }
        }

        impl $pool_name {
            /// Creates a new command pool.
            #[cfg_attr(feature = "profiling", profiling::function)]
            pub(crate) unsafe fn new(
                context: Arc<Context>,
                family_index: u32,
                id: u64,
            ) -> Result<Self> {
                let command_pool_info =
                    vk::CommandPoolCreateInfoBuilder::new().queue_family_index(family_index);

                let raw = context
                    .device
                    .create_command_pool(&command_pool_info, None)
                    .map_err(|err| {
                        #[cfg(feature = "tracing")]
                        error!("Unable to create a command pool: {}", err);
                        AscheError::VkResult(err)
                    })?;

                context.set_object_name(
                    &format!("Command Pool {} {}", $queue_type, id),
                    vk::ObjectType::COMMAND_POOL,
                    raw.0,
                )?;

                Ok(Self {
                    context,
                    raw,
                    queue_type: $queue_type,
                    command_buffer_counter: 0,
                })
            }

            /// The raw Vulkan command pool handle.
            #[inline]
            pub fn raw(&self) -> vk::CommandPool {
                self.raw
            }

            /// Creates a new command buffer.
            #[cfg_attr(feature = "profiling", profiling::function)]
            pub unsafe fn create_command_buffer(
                &mut self,
                wait_semaphores: &[CommandBufferSemaphore],
                signal_semaphores: &[CommandBufferSemaphore],
            ) -> Result<$buffer_name> {
                let info = vk::CommandBufferAllocateInfoBuilder::new()
                    .command_pool(self.raw)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                let command_buffers = self
                    .context
                    .device
                    .allocate_command_buffers(&info)
                    .map_err(|err| {
                        #[cfg(feature = "tracing")]
                        error!("Unable to allocate a command buffer: {}", err);
                        AscheError::VkResult(err)
                    })?;

                #[allow(clippy::as_conversions)]
                self.context.set_object_name(
                    &format!(
                        "Command Buffer {} {}",
                        self.queue_type, self.command_buffer_counter
                    ),
                    vk::ObjectType::COMMAND_BUFFER,
                    command_buffers[0].0 as u64,
                )?;

                let wait_semaphores: Vec<CommandBufferSemaphoreInner> =
                    wait_semaphores.iter().map(|s| s.into()).collect();
                let signal_semaphores: Vec<CommandBufferSemaphoreInner> =
                    signal_semaphores.iter().map(|s| s.into()).collect();

                let command_buffer = $buffer_name::new(
                    self.context.clone(),
                    self.raw,
                    command_buffers[0],
                    wait_semaphores,
                    signal_semaphores,
                );

                self.command_buffer_counter += 1;

                Ok(command_buffer)
            }

            /// Resets a command pool.
            #[cfg_attr(feature = "profiling", profiling::function)]
            pub unsafe fn reset(&self) -> Result<()> {
                self.context
                    .device
                    .reset_command_pool(self.raw, None)
                    .map_err(|err| {
                        #[cfg(feature = "tracing")]
                        error!("Unable to reset a command pool: {}", err);
                        AscheError::VkResult(err)
                    })?;

                Ok(())
            }
        }
    };
}

impl_command_pool!(
    #[doc = "A command pool for the compute queue."]
    ComputeCommandPool => ComputeCommandBuffer, QueueType::Compute
);

impl_command_pool!(
    #[doc = "A command pool for the graphics queue."]
    GraphicsCommandPool => GraphicsCommandBuffer, QueueType::Graphics
);

impl_command_pool!(
    #[doc = "A command pool for the transfer queue."]
    TransferCommandPool => TransferCommandBuffer, QueueType::Transfer
);

macro_rules! impl_command_buffer {
    (
        #[doc = $doc:expr]
        $buffer_name:ident => $encoder_name:ident
    ) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $buffer_name {
            raw: vk::CommandBuffer,
            pub(crate) wait_semaphores: Vec<CommandBufferSemaphoreInner>,
            pub(crate) signal_semaphores: Vec<CommandBufferSemaphoreInner>,
            pool: vk::CommandPool,
            context: Arc<Context>,
        }

        impl $buffer_name {
            pub(crate) fn new(
                context: Arc<Context>,
                pool: vk::CommandPool,
                buffer: vk::CommandBuffer,
                wait_semaphores: Vec<CommandBufferSemaphoreInner>,
                signal_semaphores: Vec<CommandBufferSemaphoreInner>,
            ) -> Self {
                Self {
                    context,
                    pool,
                    raw: buffer,
                    wait_semaphores,
                    signal_semaphores,
                }
            }

            /// The raw Vulkan command buffer handle.
            #[inline]
            pub fn raw(&self) -> vk::CommandBuffer {
                self.raw
            }

            /// Begins to record the command buffer. Encoder will finish recording on drop.
            #[cfg_attr(feature = "profiling", profiling::function)]
            pub unsafe fn record(&self) -> Result<$encoder_name> {
                let encoder = $encoder_name {
                    context: &self.context,
                    buffer: self.raw,
                };
                encoder.begin()?;
                Ok(encoder)
            }

            /// Sets the wait and signal semaphores of the command buffer.
            #[cfg_attr(feature = "profiling", profiling::function)]
            pub fn set_semaphores(
                &mut self,
                wait_semaphores: &[CommandBufferSemaphore],
                signal_semaphores: &[CommandBufferSemaphore],
            ) {
                self.wait_semaphores = wait_semaphores.iter().map(|s| s.into()).collect();
                self.signal_semaphores = signal_semaphores.iter().map(|s| s.into()).collect();
            }
        }

        impl Drop for $buffer_name {
            fn drop(&mut self) {
                unsafe {
                    self.context
                        .device
                        .free_command_buffers(self.pool, &[self.raw])
                };
            }
        }
    };
}

impl_command_buffer!(
    #[doc = "A command buffer for the compute queue. Command buffer need to be reset using the parent pool."]
    ComputeCommandBuffer => ComputeCommandEncoder
);

impl_command_buffer!(
    #[doc = "A command buffer for the compute queue. Command buffer need to be reset using the parent pool."]
    GraphicsCommandBuffer => GraphicsCommandEncoder
);

impl_command_buffer!(
    #[doc = "A command buffer for the transfer queue. Command buffer need to be reset using the parent pool."]
    TransferCommandBuffer => TransferCommandEncoder
);

/// Implements common command between all queues.
pub trait CommonCommands {
    /// Copies data between two buffer.
    unsafe fn copy_buffer(
        &self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        src_offset: vk::DeviceSize,
        dst_offset: vk::DeviceSize,
        size: vk::DeviceSize,
    );

    /// Copies data from a buffer to an image.
    unsafe fn copy_buffer_to_image(
        &self,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        region: vk::BufferImageCopyBuilder,
    );

    /// Insert a memory dependency.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier2KHR.html)"]
    unsafe fn pipeline_barrier2(&self, dependency_info: &vk::DependencyInfoKHR);

    /// Open a command buffer debug label region.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBeginDebugUtilsLabelEXT.html)"]
    unsafe fn begin_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()>;

    /// Close a command buffer label region.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdEndDebugUtilsLabelEXT.html)"]
    unsafe fn end_debug_utils_label(&self);

    /// Insert a label into a command buffer.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdInsertDebugUtilsLabelEXT.html)"]
    unsafe fn insert_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()>;
}

/// Used to encode command for a compute command buffer.
#[derive(Debug)]
pub struct ComputeCommandEncoder<'a> {
    context: &'a Context,
    buffer: vk::CommandBuffer,
}

impl<'a> CommonCommands for ComputeCommandEncoder<'a> {
    /// Copies data between two buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn copy_buffer(
        &self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        src_offset: vk::DeviceSize,
        dst_offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        copy_buffer(
            self.context,
            self.buffer,
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            size,
        )
    }

    /// Copies data from a buffer to an image.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn copy_buffer_to_image(
        &self,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        region: vk::BufferImageCopyBuilder,
    ) {
        copy_buffer_to_image(
            self.context,
            self.buffer,
            src_buffer,
            dst_image,
            dst_image_layout,
            region,
        )
    }

    /// Insert a memory dependency.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn pipeline_barrier2(&self, dependency_info: &vk::DependencyInfoKHR) {
        pipeline_barrier2(self.context, self.buffer, dependency_info);
    }

    /// Insert a memory dependency.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn begin_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()> {
        begin_debug_utils_label(self.context, self.buffer, label, color)
    }

    /// Close a command buffer label region.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn end_debug_utils_label(&self) {
        end_debug_utils_label(self.context, self.buffer);
    }

    /// Insert a label into a command buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn insert_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()> {
        insert_debug_utils_label(self.context, self.buffer, label, color)
    }
}

impl<'a> ComputeCommandEncoder<'a> {
    /// Begins a command buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn begin(&self) -> Result<()> {
        begin(self.context, self.buffer)
    }

    /// Ends a command buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn end(&self) -> Result<()> {
        end(self.context, self.buffer)
    }

    /// Binds a pipeline.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn bind_pipeline(&self, compute_pipeline: &ComputePipeline) {
        bind_pipeline(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::COMPUTE,
            compute_pipeline.raw,
        )
    }

    /// Binds descriptor sets.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn bind_descriptor_sets(
        &self,
        layout: vk::PipelineLayout,
        set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        bind_descriptor_sets(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            set,
            descriptor_sets,
            dynamic_offsets,
        )
    }

    /// Update the values of push constants.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPushConstants.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn push_constants(
        &self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) -> Result<()> {
        push_constants(
            self.context,
            self.buffer,
            layout,
            stage_flags,
            offset,
            constants,
        )
    }

    /// Dispatch compute work items.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDispatch.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        self.context
            .device
            .cmd_dispatch(self.buffer, group_count_x, group_count_y, group_count_z);
    }

    /// Dispatch compute work items.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDispatchBase.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn dispatch_base(
        &self,
        base_group_x: u32,
        base_group_y: u32,
        base_group_z: u32,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        self.context.device.cmd_dispatch_base(
            self.buffer,
            base_group_x,
            base_group_y,
            base_group_z,
            group_count_x,
            group_count_y,
            group_count_z,
        );
    }

    /// Dispatch compute work items using indirect parameters.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDispatchIndirect.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn dispatch_indirect(&self, buffer: vk::Buffer, offset: vk::DeviceSize) {
        self.context
            .device
            .cmd_dispatch_indirect(self.buffer, buffer, offset);
    }

    /// Build acceleration structures with some parameters provided on the device.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBuildAccelerationStructuresIndirectKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn build_acceleration_structures(
        &self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHRBuilder],
        build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ) {
        #[allow(clippy::as_conversions)]
        let build_range_infos = build_range_infos
            .iter()
            .map(|r| r as *const vk::AccelerationStructureBuildRangeInfoKHR)
            .collect::<Vec<*const vk::AccelerationStructureBuildRangeInfoKHR>>();

        self.context.device.cmd_build_acceleration_structures_khr(
            self.buffer,
            infos,
            &build_range_infos,
        )
    }

    /// Build an acceleration structure with some parameters provided on the device.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBuildAccelerationStructuresIndirectKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn build_acceleration_structures_indirect(
        &self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHRBuilder],
        indirect_device_addresses: &[vk::DeviceAddress],
        indirect_strides: &[u32],
        max_primitive_counts: &[*const u32],
    ) {
        self.context
            .device
            .cmd_build_acceleration_structures_indirect_khr(
                self.buffer,
                infos,
                indirect_device_addresses,
                indirect_strides,
                max_primitive_counts,
            );
    }

    /// Copy an acceleration structure.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdCopyAccelerationStructureKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn copy_acceleration_structure(&self, info: &vk::CopyAccelerationStructureInfoKHR) {
        self.context
            .device
            .cmd_copy_acceleration_structure_khr(self.buffer, info);
    }

    /// Copy an acceleration structure to device memory.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdCopyAccelerationStructureToMemoryKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn copy_acceleration_structure_to_memory(
        &self,
        info: &vk::CopyAccelerationStructureToMemoryInfoKHR,
    ) {
        self.context
            .device
            .cmd_copy_acceleration_structure_to_memory_khr(self.buffer, info);
    }

    /// Copy device memory to an acceleration structure.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdCopyMemoryToAccelerationStructureKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn copy_memory_to_acceleration_structure(
        &self,
        info: &vk::CopyMemoryToAccelerationStructureInfoKHR,
    ) {
        self.context
            .device
            .cmd_copy_memory_to_acceleration_structure_khr(self.buffer, info);
    }

    /// Write acceleration structure result parameters to query results.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdWriteAccelerationStructuresPropertiesKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn write_acceleration_structures_properties(
        &self,
        acceleration_structures: &[vk::AccelerationStructureKHR],
        query_type: vk::QueryType,
        query_pool: vk::QueryPool,
        first_query: u32,
    ) {
        self.context
            .device
            .cmd_write_acceleration_structures_properties_khr(
                self.buffer,
                acceleration_structures,
                query_type,
                query_pool,
                first_query,
            );
    }

    /// Reset queries in a query pool.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdResetQueryPool.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn reset_query_pool(
        &self,
        query_pool: vk::QueryPool,
        first_query: u32,
        query_count: u32,
    ) {
        self.context
            .device
            .cmd_reset_query_pool(self.buffer, query_pool, first_query, query_count);
    }

    /// Clear regions of a color image.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdClearColorImage.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn clear_color_image(
        &self,
        image: vk::Image,
        image_layout: vk::ImageLayout,
        color: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRangeBuilder],
    ) {
        self.context
            .device
            .cmd_clear_color_image(self.buffer, image, image_layout, color, ranges);
    }
}

impl<'a> Drop for ComputeCommandEncoder<'a> {
    fn drop(&mut self) {
        unsafe {
            self.end()
                .expect("cant' finish recording the compute command buffer");
        }
    }
}

/// Used to encode command for a graphics command buffer.
#[derive(Debug)]
pub struct GraphicsCommandEncoder<'a> {
    context: &'a Context,
    buffer: vk::CommandBuffer,
}

impl<'a> CommonCommands for GraphicsCommandEncoder<'a> {
    /// Copies data between two buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn copy_buffer(
        &self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        src_offset: vk::DeviceSize,
        dst_offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        copy_buffer(
            self.context,
            self.buffer,
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            size,
        )
    }

    /// Copies data from a buffer to an image.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn copy_buffer_to_image(
        &self,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        region: vk::BufferImageCopyBuilder,
    ) {
        copy_buffer_to_image(
            self.context,
            self.buffer,
            src_buffer,
            dst_image,
            dst_image_layout,
            region,
        )
    }

    /// Insert a memory dependency.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn pipeline_barrier2(&self, dependency_info: &vk::DependencyInfoKHR) {
        pipeline_barrier2(self.context, self.buffer, dependency_info);
    }

    /// Insert a memory dependency.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn begin_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()> {
        begin_debug_utils_label(self.context, self.buffer, label, color)
    }

    /// Close a command buffer label region.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn end_debug_utils_label(&self) {
        end_debug_utils_label(self.context, self.buffer);
    }

    /// Insert a label into a command buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn insert_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()> {
        insert_debug_utils_label(self.context, self.buffer, label, color)
    }
}

impl<'a> GraphicsCommandEncoder<'a> {
    /// Begins a command buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn begin(&self) -> Result<()> {
        begin(self.context, self.buffer)
    }

    /// Ends a command buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn end(&self) -> Result<()> {
        end(self.context, self.buffer)
    }

    /// Returns a render pass encoder. Drop once finished recording.
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn begin_render_pass(
        &self,
        swapchain: &mut Swapchain,
        render_pass: &RenderPass,
        color_attachments: &[RenderPassColorAttachmentDescriptor],
        depth_attachment: Option<RenderPassDepthAttachmentDescriptor>,
        extent: vk::Extent2D,
    ) -> Result<RenderPassEncoder> {
        let encoder = RenderPassEncoder {
            context: self.context,
            buffer: self.buffer,
        };

        let framebuffer = swapchain.next_framebuffer(
            render_pass,
            color_attachments,
            &depth_attachment,
            extent,
        )?;

        let mut clear_values: Vec<ClearValue> = Vec::with_capacity(6);

        color_attachments.iter().for_each(|x| {
            if let Some(clear_value) = x.clear_value {
                clear_values.push(clear_value)
            }
        });
        depth_attachment.iter().for_each(|x| {
            if let Some(clear_value) = x.clear_value {
                clear_values.push(clear_value)
            }
        });

        encoder.begin(render_pass.raw, framebuffer, &clear_values, extent);

        Ok(encoder)
    }

    /// Sets the viewport and the scissor rectangle.
    #[allow(clippy::as_conversions)]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn set_viewport_and_scissor(&self, rect: vk::Rect2DBuilder) {
        let viewport = vk::ViewportBuilder::new()
            .x(rect.offset.x as f32)
            .y(rect.offset.y as f32)
            .width(rect.extent.width as f32)
            .height(rect.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        self.context
            .device
            .cmd_set_viewport(self.buffer, 0, &[viewport]);
        self.context.device.cmd_set_scissor(self.buffer, 0, &[rect]);
    }

    /// Binds a pipeline.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn bind_pipeline(&self, graphics_pipeline: &GraphicsPipeline) {
        bind_pipeline(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::GRAPHICS,
            graphics_pipeline.raw,
        )
    }

    /// Binds a raytracing pipeline.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn bind_raytrace_pipeline(&self, raytracing_pipeline: &RayTracingPipeline) {
        bind_pipeline(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            raytracing_pipeline.raw,
        )
    }

    /// Binds descriptor sets.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn bind_descriptor_set(
        &self,
        pipeline_bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        bind_descriptor_sets(
            self.context,
            self.buffer,
            pipeline_bind_point,
            layout,
            set,
            descriptor_sets,
            dynamic_offsets,
        )
    }

    /// Update the values of push constants.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPushConstants.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn push_constants(
        &self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) -> Result<()> {
        push_constants(
            self.context,
            self.buffer,
            layout,
            stage_flags,
            offset,
            constants,
        )
    }

    /// Set the dynamic stack size for a ray tracing pipeline
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdSetRayTracingPipelineStackSizeKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn set_ray_tracing_pipeline_stack_size(&self, stack_size: u32) {
        self.context
            .device
            .cmd_set_ray_tracing_pipeline_stack_size_khr(self.buffer, stack_size);
    }

    /// Initialize an indirect ray tracing dispatch.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdTraceRaysIndirectKHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn trace_rays_indirect_khr(
        &self,
        raygen_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        miss_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        hit_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        callable_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        indirect_device_address: vk::DeviceAddress,
    ) {
        self.context.device.cmd_trace_rays_indirect_khr(
            self.buffer,
            raygen_shader_binding_table,
            miss_shader_binding_table,
            hit_shader_binding_table,
            callable_shader_binding_table,
            indirect_device_address,
        );
    }

    /// Initialize a ray tracing dispatch.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdTraceRaysKHR.html)"]
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn trace_rays_khr(
        &self,
        raygen_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        miss_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        hit_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        callable_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        width: u32,
        height: u32,
        depth: u32,
    ) {
        self.context.device.cmd_trace_rays_khr(
            self.buffer,
            raygen_shader_binding_table,
            miss_shader_binding_table,
            hit_shader_binding_table,
            callable_shader_binding_table,
            width,
            height,
            depth,
        );
    }

    /// Reset queries in a query pool.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdResetQueryPool.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn reset_query_pool(
        &self,
        query_pool: vk::QueryPool,
        first_query: u32,
        query_count: u32,
    ) {
        self.context
            .device
            .cmd_reset_query_pool(self.buffer, query_pool, first_query, query_count);
    }

    /// Clear regions of a color image.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdClearColorImage.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn clear_color_image(
        &self,
        image: vk::Image,
        image_layout: vk::ImageLayout,
        color: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRangeBuilder],
    ) {
        self.context
            .device
            .cmd_clear_color_image(self.buffer, image, image_layout, color, ranges);
    }
}

impl<'a> Drop for GraphicsCommandEncoder<'a> {
    fn drop(&mut self) {
        unsafe {
            self.end()
                .expect("cant' finish recording the graphics command buffer");
        }
    }
}

/// Used to encode command for a transfer command buffer.
#[derive(Debug)]
pub struct TransferCommandEncoder<'a> {
    context: &'a Context,
    buffer: vk::CommandBuffer,
}

impl<'a> CommonCommands for TransferCommandEncoder<'a> {
    /// Copies data between two buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn copy_buffer(
        &self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        src_offset: vk::DeviceSize,
        dst_offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        copy_buffer(
            self.context,
            self.buffer,
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            size,
        )
    }

    /// Copies data from a buffer to an image.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn copy_buffer_to_image(
        &self,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        region: vk::BufferImageCopyBuilder,
    ) {
        copy_buffer_to_image(
            self.context,
            self.buffer,
            src_buffer,
            dst_image,
            dst_image_layout,
            region,
        )
    }

    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier2KHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn pipeline_barrier2(&self, dependency_info: &vk::DependencyInfoKHR) {
        pipeline_barrier2(self.context, self.buffer, dependency_info);
    }

    /// Insert a memory dependency.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn begin_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()> {
        begin_debug_utils_label(self.context, self.buffer, label, color)
    }

    /// Close a command buffer label region.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn end_debug_utils_label(&self) {
        end_debug_utils_label(self.context, self.buffer);
    }

    /// Insert a label into a command buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn insert_debug_utils_label(&self, label: &str, color: [f32; 4]) -> Result<()> {
        insert_debug_utils_label(self.context, self.buffer, label, color)
    }
}

impl<'a> TransferCommandEncoder<'a> {
    /// Begins a command buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn begin(&self) -> Result<()> {
        begin(self.context, self.buffer)
    }

    /// Ends a command buffer.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn end(&self) -> Result<()> {
        end(self.context, self.buffer)
    }
}

impl<'a> Drop for TransferCommandEncoder<'a> {
    fn drop(&mut self) {
        unsafe {
            self.end()
                .expect("cant' finish recording the transfer command buffer");
        }
    }
}

/// Used to encode render pass commands of a command buffer.
#[derive(Debug)]
pub struct RenderPassEncoder<'a> {
    context: &'a Context,
    buffer: vk::CommandBuffer,
}

impl<'a> Drop for RenderPassEncoder<'a> {
    fn drop(&mut self) {
        unsafe { self.context.device.cmd_end_render_pass(self.buffer) };
    }
}

impl<'a> RenderPassEncoder<'a> {
    /// Begins a render pass.
    #[cfg_attr(feature = "profiling", profiling::function)]
    unsafe fn begin(
        &self,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        clear_values: &[vk::ClearValue],
        extent: vk::Extent2D,
    ) {
        let create_info = vk::RenderPassBeginInfoBuilder::new()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .clear_values(clear_values)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            });
        let contents = vk::SubpassContents::INLINE;

        self.context
            .device
            .cmd_begin_render_pass(self.buffer, &create_info, contents);
    }

    /// Bind a pipeline object to a command buffer.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn bind_pipeline(&self, graphics_pipeline: &GraphicsPipeline) {
        bind_pipeline(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::GRAPHICS,
            graphics_pipeline.raw,
        )
    }

    /// Binds descriptor sets.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn bind_descriptor_sets(
        &self,
        layout: vk::PipelineLayout,
        set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        bind_descriptor_sets(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            set,
            descriptor_sets,
            dynamic_offsets,
        )
    }

    /// Bind an index buffer to a command buffer.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindIndexBuffer.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn bind_index_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        self.context
            .device
            .cmd_bind_index_buffer(self.buffer, buffer, offset, index_type);
    }

    /// Bind vertex buffers to a command buffer.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindVertexBuffers.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn bind_vertex_buffers(
        &self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[u64],
    ) {
        self.context
            .device
            .cmd_bind_vertex_buffers(self.buffer, first_binding, buffers, offsets);
    }

    /// Update the values of push constants.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPushConstants.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn push_constants(
        &self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) -> Result<()> {
        push_constants(
            self.context,
            self.buffer,
            layout,
            stage_flags,
            offset,
            constants,
        )
    }

    /// Insert a memory dependency.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier2KHR.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn pipeline_barrier2(&self, dependency_info: &vk::DependencyInfoKHR) {
        pipeline_barrier2(self.context, self.buffer, dependency_info);
    }

    /// Draws primitives.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDraw.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.context.device.cmd_draw(
            self.buffer,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );
    }

    /// Issue an indexed draw into a command buffer.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndexed.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.context.device.cmd_draw_indexed(
            self.buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    /// Perform an indexed indirect draw.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndexedIndirect.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn draw_indexed_indirect(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        self.context.device.cmd_draw_indexed_indirect(
            self.buffer,
            buffer,
            offset,
            draw_count,
            stride,
        );
    }

    /// Perform an indexed indirect draw with the draw count sourced from a buffer.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndexedIndirect.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn draw_indexed_indirect_count(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_buffer_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        self.context.device.cmd_draw_indexed_indirect_count(
            self.buffer,
            buffer,
            offset,
            count_buffer,
            count_buffer_offset,
            max_draw_count,
            stride,
        );
    }

    /// Perform an indexed indirect draw with the draw count sourced from a buffer.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndexedIndirect.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn draw_indirect(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        self.context
            .device
            .cmd_draw_indirect(self.buffer, buffer, offset, draw_count, stride);
    }

    /// Perform an indexed indirect draw with the draw count sourced from a buffer.
    #[doc = "[Vulkan Manual Page](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndirectCount.html)"]
    #[cfg_attr(feature = "profiling", profiling::function)]
    pub unsafe fn draw_indirect_count(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_buffer_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        self.context.device.cmd_draw_indirect_count(
            self.buffer,
            buffer,
            offset,
            count_buffer,
            count_buffer_offset,
            max_draw_count,
            stride,
        );
    }
}

#[inline]
unsafe fn begin(context: &Context, buffer: vk::CommandBuffer) -> Result<()> {
    let info = vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    context
        .device
        .begin_command_buffer(buffer, &info)
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to begin a command buffer: {}", err);
            AscheError::VkResult(err)
        })?;

    Ok(())
}

#[inline]
unsafe fn end(context: &Context, buffer: vk::CommandBuffer) -> Result<()> {
    context.device.end_command_buffer(buffer).map_err(|err| {
        #[cfg(feature = "tracing")]
        error!("Unable to end a command buffer: {}", err);
        AscheError::VkResult(err)
    })?;

    Ok(())
}

#[inline]
unsafe fn bind_pipeline(
    context: &Context,
    buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    context
        .device
        .cmd_bind_pipeline(buffer, pipeline_bind_point, pipeline)
}

#[inline]
unsafe fn bind_descriptor_sets(
    context: &Context,
    buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    set: u32,
    descriptor_sets: &[vk::DescriptorSet],
    dynamic_offsets: &[u32],
) {
    context.device.cmd_bind_descriptor_sets(
        buffer,
        pipeline_bind_point,
        layout,
        set,
        descriptor_sets,
        dynamic_offsets,
    )
}

#[inline]
unsafe fn copy_buffer(
    context: &Context,
    buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    src_offset: vk::DeviceSize,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
) {
    let region = vk::BufferCopyBuilder::new()
        .dst_offset(dst_offset)
        .src_offset(src_offset)
        .size(size);
    context
        .device
        .cmd_copy_buffer(buffer, src_buffer, dst_buffer, &[region]);
}

#[inline]
unsafe fn copy_buffer_to_image(
    context: &Context,
    buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region: vk::BufferImageCopyBuilder,
) {
    context.device.cmd_copy_buffer_to_image(
        buffer,
        src_buffer,
        dst_image,
        dst_image_layout,
        &[region],
    );
}

#[inline]
unsafe fn push_constants(
    context: &Context,
    buffer: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    stage_flags: vk::ShaderStageFlags,
    offset: u32,
    constants: &[u8],
) -> Result<()> {
    let size: u32 = constants.len().try_into()?;

    #[allow(clippy::as_conversions)]
    context.device.cmd_push_constants(
        buffer,
        layout,
        stage_flags,
        offset,
        size,
        constants.as_ptr() as *mut c_void,
    );

    Ok(())
}

#[inline]
unsafe fn pipeline_barrier2(
    context: &Context,
    command_buffer: vk::CommandBuffer,
    dependency_info: &vk::DependencyInfoKHR,
) {
    context
        .device
        .cmd_pipeline_barrier2_khr(command_buffer, dependency_info);
}

#[inline]
unsafe fn begin_debug_utils_label(
    context: &Context,
    command_buffer: vk::CommandBuffer,
    label: &str,
    color: [f32; 4],
) -> Result<()> {
    let label = CString::new(label.to_string())?;

    context.device.cmd_begin_debug_utils_label_ext(
        command_buffer,
        &vk::DebugUtilsLabelEXTBuilder::new()
            .label_name(label.as_c_str())
            .color(color),
    );

    Ok(())
}

#[inline]
unsafe fn end_debug_utils_label(context: &Context, command_buffer: vk::CommandBuffer) {
    context.device.cmd_end_debug_utils_label_ext(command_buffer);
}

#[inline]
unsafe fn insert_debug_utils_label(
    context: &Context,
    command_buffer: vk::CommandBuffer,
    label: &str,
    color: [f32; 4],
) -> Result<()> {
    let label = CString::new(label.to_string())?;

    context.device.cmd_insert_debug_utils_label_ext(
        command_buffer,
        &vk::DebugUtilsLabelEXTBuilder::new()
            .label_name(label.as_c_str())
            .color(color),
    );

    Ok(())
}
