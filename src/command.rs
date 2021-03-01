//! Implements command pools and command buffers.

use std::ffi::c_void;
use std::sync::Arc;

use arrayvec::ArrayVec;
use erupt::vk;

use crate::context::Context;
use crate::descriptor::DescriptorSet;
use crate::semaphore::TimelineSemaphore;
use crate::{
    Buffer, ComputePipeline, GraphicsPipeline, Image, PipelineLayout, QueueType, RenderPass,
    RenderPassColorAttachmentDescriptor, RenderPassDepthAttachmentDescriptor, Result,
};

macro_rules! impl_command_pool {
    (
        #[doc = $doc:expr]
        $pool_name:ident => $buffer_name:ident, $queue_type:expr
    ) => {
        #[doc = $doc]
        pub struct $pool_name {
            inner: CommandPool,
        }

        impl $pool_name {
            /// Creates a new command pool.
            pub(crate) fn new(context: Arc<Context>, family_index: u32, id: u64) -> Result<Self> {
                let inner = CommandPool::new(context, $queue_type, family_index, id)?;
                Ok(Self { inner })
            }

            /// Creates a new command buffer.
            pub fn create_command_buffer(
                &mut self,
                timeline_semaphore: &TimelineSemaphore,
                timeline_wait_value: u64,
                timeline_signal_value: u64,
            ) -> Result<$buffer_name> {
                let inner = self.inner
                    .create_command_buffer(timeline_semaphore.raw, timeline_wait_value, timeline_signal_value)?;
                Ok($buffer_name { inner })
            }

            /// Resets a command pool.
            pub fn reset(&self) -> Result<()> {
                self.inner.reset()
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

/// A wrapped command pool.
struct CommandPool {
    raw: vk::CommandPool,
    queue_type: QueueType,
    command_buffer_counter: u64,
    context: Arc<Context>,
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_command_pool(Some(self.raw), None);
        };
    }
}

impl CommandPool {
    /// Creates a new command pool.
    pub(crate) fn new(
        context: Arc<Context>,
        queue_type: QueueType,
        family_index: u32,
        id: u64,
    ) -> Result<Self> {
        let command_pool_info =
            vk::CommandPoolCreateInfoBuilder::new().queue_family_index(family_index);

        let raw = unsafe {
            context
                .device
                .create_command_pool(&command_pool_info, None, None)
                .result()?
        };

        context.set_object_name(
            &format!("Command Pool {} {}", queue_type, id),
            vk::ObjectType::COMMAND_POOL,
            raw.0,
        )?;

        Ok(Self {
            context,
            raw,
            queue_type,
            command_buffer_counter: 0,
        })
    }

    /// Creates a new command buffer.
    #[inline]
    pub fn create_command_buffer(
        &mut self,
        timeline_semaphore: vk::Semaphore,
        timeline_wait_value: u64,
        timeline_signal_value: u64,
    ) -> Result<CommandBuffer> {
        let info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(self.raw)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe {
            self.context
                .device
                .allocate_command_buffers(&info)
                .result()?
        };

        self.context.set_object_name(
            &format!(
                "Command Buffer {} {}",
                self.queue_type, self.command_buffer_counter
            ),
            vk::ObjectType::COMMAND_BUFFER,
            command_buffers[0].0 as u64,
        )?;

        let command_buffer = CommandBuffer::new(
            self.context.clone(),
            self.raw,
            command_buffers[0],
            timeline_semaphore,
            timeline_wait_value,
            timeline_signal_value,
        );

        self.command_buffer_counter += 1;

        Ok(command_buffer)
    }

    /// Resets a command pool.
    #[inline]
    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.context
                .device
                .reset_command_pool(self.raw, None)
                .result()?
        };

        Ok(())
    }
}

macro_rules! impl_command_buffer {
    (
        #[doc = $doc:expr]
        $buffer_name:ident => $encoder_name:ident
    ) => {

        #[doc = $doc]
        pub struct $buffer_name {
            pub(crate) inner: CommandBuffer,
        }

        impl $buffer_name {
            /// Records the command buffer actions with the help of an encoder.
            pub fn record<F>(&self, exec: F) -> Result<()>
            where
                F: Fn(&$encoder_name) -> Result<()>,
            {
                let encoder = $encoder_name {
                    context: &self.inner.context,
                    buffer: self.inner.buffer
                };

                encoder.begin()?;
                exec(&encoder)?;
                encoder.end()?;

                Ok(())
            }

            /// Sets the timeline semaphore of a command buffer.
            pub fn set_timeline_semaphore(&mut self, timeline_semaphore: TimelineSemaphore, wait_value: u64, signal_value: u64) {
                self.inner.set_timeline_semaphore(timeline_semaphore.raw, wait_value, signal_value)
            }
        }
    }
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

/// A wrapped command buffer.
pub(crate) struct CommandBuffer {
    pool: vk::CommandPool,
    pub(crate) buffer: vk::CommandBuffer,
    pub(crate) timeline_semaphore: vk::Semaphore,
    pub(crate) timeline_wait_value: u64,
    pub(crate) timeline_signal_value: u64,
    context: Arc<Context>,
}

impl CommandBuffer {
    pub(crate) fn new(
        context: Arc<Context>,
        pool: vk::CommandPool,
        buffer: vk::CommandBuffer,
        timeline_semaphore: vk::Semaphore,
        timeline_wait_value: u64,
        timeline_signal_value: u64,
    ) -> Self {
        Self {
            context,
            pool,
            buffer,
            timeline_semaphore,
            timeline_wait_value,
            timeline_signal_value,
        }
    }

    #[inline]
    fn set_timeline_semaphore(
        &mut self,
        timeline_semaphore: vk::Semaphore,
        wait_value: u64,
        signal_value: u64,
    ) {
        self.timeline_semaphore = timeline_semaphore;
        self.timeline_wait_value = wait_value;
        self.timeline_signal_value = signal_value;
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .free_command_buffers(self.pool, &[self.buffer])
        };
    }
}

/// Used to encode command for a compute command buffer.
pub struct ComputeCommandEncoder<'a> {
    context: &'a Context,
    buffer: vk::CommandBuffer,
}

impl<'a> ComputeCommandEncoder<'a> {
    /// Begins a command buffer.
    fn begin(&self) -> Result<()> {
        begin(self.context, self.buffer)
    }

    /// Ends a command buffer.
    fn end(&self) -> Result<()> {
        end(self.context, self.buffer)
    }

    /// Copies data between two buffer.
    pub fn copy_buffer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        copy_buffer(
            self.context,
            self.buffer,
            src_buffer.raw,
            dst_buffer.raw,
            src_offset,
            dst_offset,
            size,
        )
    }

    /// Copies data from a buffer to an image.
    pub fn copy_buffer_to_image(
        &self,
        src_buffer: &Buffer,
        dst_image: &Image,
        dst_image_layout: vk::ImageLayout,
        region: vk::BufferImageCopyBuilder,
    ) {
        copy_buffer_to_image(
            self.context,
            self.buffer,
            src_buffer.raw,
            dst_image.raw,
            dst_image_layout,
            region,
        )
    }

    /// Binds a pipeline.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html
    pub fn bind_pipeline(&self, compute_pipeline: &ComputePipeline) {
        bind_pipeline(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::COMPUTE,
            compute_pipeline.raw,
        )
    }

    /// Binds a descriptor set.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html
    pub fn bind_descriptor_set(
        &self,
        layout: &PipelineLayout,
        set: u32,
        descriptor_set: &DescriptorSet,
        dynamic_offsets: &[u32],
    ) {
        bind_descriptor_sets(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::COMPUTE,
            layout.raw,
            set,
            descriptor_set.raw,
            dynamic_offsets,
        )
    }

    /// Update the values of push constants.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPushConstants.html
    pub fn push_constants(
        &self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        push_constants(
            self.context,
            self.buffer,
            layout,
            stage_flags,
            offset,
            constants,
        );
    }

    /// Insert a memory dependency.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier.html
    pub fn pipeline_barrier(
        &self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: Option<vk::DependencyFlags>,
        memory_barriers: &[vk::MemoryBarrierBuilder],
        buffer_memory_barriers: &[vk::BufferMemoryBarrierBuilder],
        image_memory_barriers: &[vk::ImageMemoryBarrierBuilder],
    ) {
        pipeline_barrier(
            self.context,
            self.buffer,
            src_stage_mask,
            dst_stage_mask,
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
        );
    }

    /// Dispatch compute work items.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDispatch.html
    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.context.device.cmd_dispatch(
                self.buffer,
                group_count_x,
                group_count_y,
                group_count_z,
            )
        };
    }

    /// Dispatch compute work items.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDispatchBase.html
    pub fn dispatch_base(
        &self,
        base_group_x: u32,
        base_group_y: u32,
        base_group_z: u32,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        unsafe {
            self.context.device.cmd_dispatch_base(
                self.buffer,
                base_group_x,
                base_group_y,
                base_group_z,
                group_count_x,
                group_count_y,
                group_count_z,
            )
        };
    }

    /// Dispatch compute work items using indirect parameters.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDispatchIndirect.html
    pub fn dispatch_indirect(&self, buffer: &Buffer, offset: u64) {
        unsafe {
            self.context
                .device
                .cmd_dispatch_indirect(self.buffer, buffer.raw, offset)
        };
    }
}

/// Used to encode command for a graphics command buffer.
pub struct GraphicsCommandEncoder<'a> {
    context: &'a Context,
    buffer: vk::CommandBuffer,
}

impl<'a> GraphicsCommandEncoder<'a> {
    /// Begins a command buffer.
    fn begin(&self) -> Result<()> {
        begin(self.context, self.buffer)
    }

    /// Ends a command buffer.
    fn end(&self) -> Result<()> {
        end(self.context, self.buffer)
    }

    /// Copies data between two buffer.
    pub fn copy_buffer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        copy_buffer(
            self.context,
            self.buffer,
            src_buffer.raw,
            dst_buffer.raw,
            src_offset,
            dst_offset,
            size,
        )
    }

    /// Copies data from a buffer to an image.
    pub fn copy_buffer_to_image(
        &self,
        src_buffer: &Buffer,
        dst_image: &Image,
        dst_image_layout: vk::ImageLayout,
        region: vk::BufferImageCopyBuilder,
    ) {
        copy_buffer_to_image(
            self.context,
            self.buffer,
            src_buffer.raw,
            dst_image.raw,
            dst_image_layout,
            region,
        )
    }

    /// Returns a render pass encoder. Drop once finished recording.
    pub fn begin_render_pass(
        &self,
        render_pass: &RenderPass,
        color_attachments: &[&RenderPassColorAttachmentDescriptor],
        depth_attachment: Option<&RenderPassDepthAttachmentDescriptor>,
        extent: vk::Extent2D,
    ) -> Result<RenderPassEncoder> {
        let encoder = RenderPassEncoder {
            context: self.context,
            buffer: self.buffer,
        };

        let framebuffer = self.context.get_framebuffer(
            render_pass,
            color_attachments,
            depth_attachment,
            extent,
        )?;

        // 64 byte array on the stack.
        let mut clear_values = ArrayVec::<[vk::ClearValue; 4]>::new();
        color_attachments
            .iter()
            .map(|x| x.clear_value)
            .chain(depth_attachment.iter().map(|x| x.clear_value))
            .for_each(|x| clear_values.push(x));

        encoder.begin(render_pass.raw, framebuffer, &clear_values, extent);

        Ok(encoder)
    }

    /// Sets the viewport and the scissor rectangle.
    pub fn set_viewport_and_scissor(&self, rect: vk::Rect2DBuilder) {
        let viewport = vk::ViewportBuilder::new()
            .x(rect.offset.x as f32)
            .y(rect.offset.y as f32)
            .width(rect.extent.width as f32)
            .height(rect.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        unsafe {
            self.context
                .device
                .cmd_set_viewport(self.buffer, 0, &[viewport]);
            self.context.device.cmd_set_scissor(self.buffer, 0, &[rect]);
        };
    }

    /// Binds a pipeline.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html
    pub fn bind_pipeline(&self, graphics_pipeline: &GraphicsPipeline) {
        bind_pipeline(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::GRAPHICS,
            graphics_pipeline.raw,
        )
    }

    /// Binds a descriptor set.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html
    pub fn bind_descriptor_set(
        &self,
        layout: &PipelineLayout,
        set: u32,
        descriptor_set: &DescriptorSet,
        dynamic_offsets: &[u32],
    ) {
        bind_descriptor_sets(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::GRAPHICS,
            layout.raw,
            set,
            descriptor_set.raw,
            dynamic_offsets,
        )
    }

    /// Update the values of push constants.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPushConstants.html
    pub fn push_constants(
        &self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        push_constants(
            self.context,
            self.buffer,
            layout,
            stage_flags,
            offset,
            constants,
        );
    }

    /// Insert a memory dependency.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier.html
    pub fn pipeline_barrier(
        &self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: Option<vk::DependencyFlags>,
        memory_barriers: &[vk::MemoryBarrierBuilder],
        buffer_memory_barriers: &[vk::BufferMemoryBarrierBuilder],
        image_memory_barriers: &[vk::ImageMemoryBarrierBuilder],
    ) {
        pipeline_barrier(
            self.context,
            self.buffer,
            src_stage_mask,
            dst_stage_mask,
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
        );
    }
}

/// Used to encode command for a transfer command buffer.
pub struct TransferCommandEncoder<'a> {
    context: &'a Context,
    buffer: vk::CommandBuffer,
}

impl<'a> TransferCommandEncoder<'a> {
    /// Begins a command buffer.
    fn begin(&self) -> Result<()> {
        begin(self.context, self.buffer)
    }

    /// Ends a command buffer.
    fn end(&self) -> Result<()> {
        end(self.context, self.buffer)
    }

    /// Copies data between two buffer.
    pub fn copy_buffer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        copy_buffer(
            self.context,
            self.buffer,
            src_buffer.raw,
            dst_buffer.raw,
            src_offset,
            dst_offset,
            size,
        )
    }

    /// Copies data from a buffer to an image.
    pub fn copy_buffer_to_image(
        &self,
        src_buffer: &Buffer,
        dst_image: &Image,
        dst_image_layout: vk::ImageLayout,
        region: vk::BufferImageCopyBuilder,
    ) {
        copy_buffer_to_image(
            self.context,
            self.buffer,
            src_buffer.raw,
            dst_image.raw,
            dst_image_layout,
            region,
        )
    }

    /// Insert a memory dependency.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier.html
    pub fn pipeline_barrier(
        &self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: Option<vk::DependencyFlags>,
        memory_barriers: &[vk::MemoryBarrierBuilder],
        buffer_memory_barriers: &[vk::BufferMemoryBarrierBuilder],
        image_memory_barriers: &[vk::ImageMemoryBarrierBuilder],
    ) {
        pipeline_barrier(
            self.context,
            self.buffer,
            src_stage_mask,
            dst_stage_mask,
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
        );
    }
}

/// Used to encode render pass commands of a command buffer.
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
    fn begin(
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

        unsafe {
            self.context
                .device
                .cmd_begin_render_pass(self.buffer, &create_info, contents)
        };
    }

    /// Bind a pipeline object to a command buffer.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html
    pub fn bind_pipeline(&self, graphics_pipeline: &GraphicsPipeline) {
        bind_pipeline(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::GRAPHICS,
            graphics_pipeline.raw,
        )
    }

    /// Binds a descriptor set.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html
    pub fn bind_descriptor_set(
        &self,
        layout: &PipelineLayout,
        set: u32,
        descriptor_set: &DescriptorSet,
        dynamic_offsets: &[u32],
    ) {
        bind_descriptor_sets(
            self.context,
            self.buffer,
            vk::PipelineBindPoint::GRAPHICS,
            layout.raw,
            set,
            descriptor_set.raw,
            dynamic_offsets,
        )
    }

    /// Bind an index buffer to a command buffer.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindIndexBuffer.html
    pub fn bind_index_buffer(
        &self,
        index_buffer: vk::Buffer,
        offset: u64,
        index_type: vk::IndexType,
    ) {
        unsafe {
            self.context
                .device
                .cmd_bind_index_buffer(self.buffer, index_buffer, offset, index_type)
        };
    }

    /// Bind vertex buffers to a command buffer.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindVertexBuffers.html
    pub fn bind_vertex_buffer(
        &self,
        first_binding: u32,
        vertex_buffers: &[vk::Buffer],
        offsets: &[u64],
    ) {
        unsafe {
            self.context.device.cmd_bind_vertex_buffers(
                self.buffer,
                first_binding,
                vertex_buffers,
                offsets,
            )
        };
    }

    /// Update the values of push constants.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPushConstants.html
    pub fn push_constants(
        &self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        push_constants(
            self.context,
            self.buffer,
            layout,
            stage_flags,
            offset,
            constants,
        );
    }

    /// Insert a memory dependency.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier.html
    pub fn pipeline_barrier(
        &self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: Option<vk::DependencyFlags>,
        memory_barriers: &[vk::MemoryBarrierBuilder],
        buffer_memory_barriers: &[vk::BufferMemoryBarrierBuilder],
        image_memory_barriers: &[vk::ImageMemoryBarrierBuilder],
    ) {
        pipeline_barrier(
            self.context,
            self.buffer,
            src_stage_mask,
            dst_stage_mask,
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
        );
    }

    /// Draws primitives.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDraw.html
    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.context.device.cmd_draw(
                self.buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        };
    }

    /// Issue an indexed draw into a command buffer.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndexed.html
    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.context.device.cmd_draw_indexed(
                self.buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };
    }

    /// Perform an indexed indirect draw.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndexedIndirect.html
    pub fn draw_indexed_indirect(
        &self,
        buffer: &Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.context.device.cmd_draw_indexed_indirect(
                self.buffer,
                buffer.raw,
                offset,
                draw_count,
                stride,
            )
        };
    }

    /// Perform an indexed indirect draw with the draw count sourced from a buffer.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndexedIndirect.html
    pub fn draw_indexed_indirect_count(
        &self,
        buffer: &Buffer,
        offset: u64,
        count_buffer: &Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.context.device.cmd_draw_indexed_indirect_count(
                self.buffer,
                buffer.raw,
                offset,
                count_buffer.raw,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };
    }

    /// Perform an indexed indirect draw with the draw count sourced from a buffer.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndexedIndirect.html
    pub fn draw_indirect(&self, buffer: &Buffer, offset: u64, draw_count: u32, stride: u32) {
        unsafe {
            self.context.device.cmd_draw_indirect(
                self.buffer,
                buffer.raw,
                offset,
                draw_count,
                stride,
            )
        };
    }

    /// Perform an indexed indirect draw with the draw count sourced from a buffer.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDrawIndirectCount.html
    pub fn draw_indirect_count(
        &self,
        buffer: &Buffer,
        offset: u64,
        count_buffer: &Buffer,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.context.device.cmd_draw_indirect_count(
                self.buffer,
                buffer.raw,
                offset,
                count_buffer.raw,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };
    }
}

#[inline]
fn begin(context: &Context, buffer: vk::CommandBuffer) -> Result<()> {
    let info = vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        context
            .device
            .begin_command_buffer(buffer, &info)
            .result()?
    };

    Ok(())
}

#[inline]
fn end(context: &Context, buffer: vk::CommandBuffer) -> Result<()> {
    unsafe { context.device.end_command_buffer(buffer).result()? };

    Ok(())
}

#[inline]
fn bind_pipeline(
    context: &Context,
    buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    unsafe {
        context
            .device
            .cmd_bind_pipeline(buffer, pipeline_bind_point, pipeline)
    };
}

#[inline]
fn bind_descriptor_sets(
    context: &Context,
    buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    set: u32,
    descriptor_set: vk::DescriptorSet,
    dynamic_offsets: &[u32],
) {
    let descriptor_sets = [descriptor_set];
    unsafe {
        context.device.cmd_bind_descriptor_sets(
            buffer,
            pipeline_bind_point,
            layout,
            set,
            &descriptor_sets,
            dynamic_offsets,
        )
    };
}

#[inline]
fn copy_buffer(
    context: &Context,
    buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
) {
    let region = vk::BufferCopyBuilder::new()
        .dst_offset(dst_offset)
        .src_offset(src_offset)
        .size(size);
    unsafe {
        context
            .device
            .cmd_copy_buffer(buffer, src_buffer, dst_buffer, &[region])
    };
}

#[inline]
fn copy_buffer_to_image(
    context: &Context,
    buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region: vk::BufferImageCopyBuilder,
) {
    unsafe {
        context.device.cmd_copy_buffer_to_image(
            buffer,
            src_buffer,
            dst_image,
            dst_image_layout,
            &[region],
        )
    };
}

#[inline]
fn push_constants(
    context: &Context,
    buffer: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    stage_flags: vk::ShaderStageFlags,
    offset: u32,
    constants: &[u8],
) {
    unsafe {
        context.device.cmd_push_constants(
            buffer,
            layout,
            stage_flags,
            offset,
            constants.len() as u32,
            constants.as_ptr() as *mut c_void,
        )
    };
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn pipeline_barrier(
    context: &Context,
    command_buffer: vk::CommandBuffer,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flags: Option<vk::DependencyFlags>,
    memory_barriers: &[vk::MemoryBarrierBuilder],
    buffer_memory_barriers: &[vk::BufferMemoryBarrierBuilder],
    image_memory_barriers: &[vk::ImageMemoryBarrierBuilder],
) {
    unsafe {
        context.device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_stage_mask,
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
        )
    };
}
