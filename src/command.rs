//! Implements command pools and command buffers.

use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::Handle;

use crate::context::Context;
use crate::{Buffer, ComputePipeline, GraphicsPipeline, QueueType, RenderPass, Result};

// TODO function to set the timeline values
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
                timeline_wait_value: u64,
                timeline_signal_value: u64,
            ) -> Result<$buffer_name> {
                let inner = self.inner
                    .create_command_buffer(timeline_wait_value, timeline_signal_value)?;
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
    context: Arc<Context>,
    raw: vk::CommandPool,
    queue_type: QueueType,
    command_buffer_counter: u64,
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_command_pool(self.raw, None);
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
            vk::CommandPoolCreateInfo::builder().queue_family_index(family_index);

        let raw = unsafe {
            context
                .logical_device
                .create_command_pool(&command_pool_info, None)?
        };

        context.set_object_name(
            &format!("Command Pool {} {}", queue_type, id),
            vk::ObjectType::COMMAND_POOL,
            raw.as_raw(),
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
        timeline_wait_value: u64,
        timeline_signal_value: u64,
    ) -> Result<CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.raw)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe {
            self.context
                .logical_device
                .allocate_command_buffers(&info)?
        };

        self.context.set_object_name(
            &format!(
                "Command Buffer {} {}",
                self.queue_type, self.command_buffer_counter
            ),
            vk::ObjectType::COMMAND_BUFFER,
            command_buffers[0].as_raw(),
        )?;

        let command_buffer = CommandBuffer::new(
            self.context.clone(),
            self.raw,
            command_buffers[0],
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
                .logical_device
                .reset_command_pool(self.raw, vk::CommandPoolResetFlags::empty())?
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
                    context: self.inner.context.clone(),
                    buffer: self.inner.buffer
                };

                encoder.begin()?;
                exec(&encoder)?;
                encoder.end()?;

                Ok(())
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
    context: Arc<Context>,
    pool: vk::CommandPool,
    pub(crate) buffer: vk::CommandBuffer,
    pub(crate) timeline_wait_value: u64,
    pub(crate) timeline_signal_value: u64,
}

impl CommandBuffer {
    pub(crate) fn new(
        context: Arc<Context>,
        pool: vk::CommandPool,
        buffer: vk::CommandBuffer,
        timeline_wait_value: u64,
        timeline_signal_value: u64,
    ) -> Self {
        Self {
            context,
            pool,
            buffer,
            timeline_wait_value,
            timeline_signal_value,
        }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .free_command_buffers(self.pool, &[self.buffer])
        };
    }
}

/// Used to encode command for a compute command buffer.
pub struct ComputeCommandEncoder {
    context: Arc<Context>,
    buffer: vk::CommandBuffer,
}

impl ComputeCommandEncoder {
    /// Begins a command buffer.
    fn begin(&self) -> Result<()> {
        begin(&self.context, self.buffer)
    }

    /// Ends a command buffer.
    fn end(&self) -> Result<()> {
        end(&self.context, self.buffer)
    }

    /// Copies data between two buffer.
    pub fn cmd_copy_buffer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        cmd_copy_buffer(
            &self.context,
            self.buffer,
            src_buffer.raw,
            dst_buffer.raw,
            src_offset,
            dst_offset,
            size,
        )
    }

    /// Binds a pipeline.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html
    pub fn cmd_bind_pipeline(&self, compute_pipeline: &ComputePipeline) {
        cmd_bind_pipeline(
            &self.context,
            self.buffer,
            vk::PipelineBindPoint::COMPUTE,
            compute_pipeline.raw,
        )
    }
}

/// Used to encode command for a graphics command buffer.
pub struct GraphicsCommandEncoder {
    context: Arc<Context>,
    buffer: vk::CommandBuffer,
}

impl GraphicsCommandEncoder {
    /// Begins a command buffer.
    fn begin(&self) -> Result<()> {
        begin(&self.context, self.buffer)
    }

    /// Ends a command buffer.
    fn end(&self) -> Result<()> {
        end(&self.context, self.buffer)
    }

    /// Copies data between two buffer.
    pub fn cmd_copy_buffer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        cmd_copy_buffer(
            &self.context,
            self.buffer,
            src_buffer.raw,
            dst_buffer.raw,
            src_offset,
            dst_offset,
            size,
        )
    }

    /// Returns a render pass encoder. Drop once finished recording.
    pub fn begin_render_pass(
        &self,
        render_pass: &RenderPass,
        frame_buffer: vk::Framebuffer,
        clear_values: &[vk::ClearValue],
        render_area: vk::Rect2D,
    ) -> Result<RenderPassEncoder> {
        let encoder = RenderPassEncoder {
            context: self.context.clone(),
            buffer: self.buffer,
        };
        encoder.begin(render_pass.raw, frame_buffer, clear_values, render_area);

        Ok(encoder)
    }

    /// Sets the viewport and the scissor rectangle.
    pub fn set_viewport_and_scissor(&self, rect: vk::Rect2D) {
        unsafe {
            let viewport = vk::Viewport {
                x: rect.offset.x as f32,
                y: rect.offset.y as f32,
                width: rect.extent.width as f32,
                height: rect.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            self.context
                .logical_device
                .cmd_set_viewport(self.buffer, 0, &[viewport]);
            self.context
                .logical_device
                .cmd_set_scissor(self.buffer, 0, &[rect]);
        };
    }

    /// Binds a pipeline.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html
    pub fn cmd_bind_pipeline(&self, graphics_pipeline: &GraphicsPipeline) {
        cmd_bind_pipeline(
            &self.context,
            self.buffer,
            vk::PipelineBindPoint::GRAPHICS,
            graphics_pipeline.raw,
        )
    }
}

/// Used to encode command for a transfer command buffer.
pub struct TransferCommandEncoder {
    context: Arc<Context>,
    buffer: vk::CommandBuffer,
}

impl TransferCommandEncoder {
    /// Begins a command buffer.
    fn begin(&self) -> Result<()> {
        begin(&self.context, self.buffer)
    }

    /// Ends a command buffer.
    fn end(&self) -> Result<()> {
        end(&self.context, self.buffer)
    }

    /// Copies data between two buffer.
    pub fn cmd_copy_buffer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        cmd_copy_buffer(
            &self.context,
            self.buffer,
            src_buffer.raw,
            dst_buffer.raw,
            src_offset,
            dst_offset,
            size,
        )
    }
}

/// Used to encode render pass commands of a command buffer.
pub struct RenderPassEncoder {
    pub(crate) context: Arc<Context>,
    pub(crate) buffer: vk::CommandBuffer,
}

impl Drop for RenderPassEncoder {
    fn drop(&mut self) {
        unsafe { self.context.logical_device.cmd_end_render_pass(self.buffer) };
    }
}

impl RenderPassEncoder {
    /// Begins a render pass.
    fn begin(
        &self,
        render_pass: vk::RenderPass,
        frame_buffer: vk::Framebuffer,
        clear_values: &[vk::ClearValue],
        render_area: vk::Rect2D,
    ) {
        let create_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(frame_buffer)
            .clear_values(clear_values)
            .render_area(render_area);
        let contents = vk::SubpassContents::INLINE;

        unsafe {
            self.context
                .logical_device
                .cmd_begin_render_pass(self.buffer, &create_info, contents)
        };
    }

    /// Binds a pipeline.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html
    pub fn cmd_bind_pipeline(&self, graphics_pipeline: &GraphicsPipeline) {
        cmd_bind_pipeline(
            &self.context,
            self.buffer,
            vk::PipelineBindPoint::GRAPHICS,
            graphics_pipeline.raw,
        )
    }

    /// Binds an index buffer.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindIndexBuffer.html
    pub fn cmd_bind_index_buffer(
        &self,
        index_buffer: vk::Buffer,
        offset: u64,
        index_type: vk::IndexType,
    ) {
        cmd_bind_index_buffer(&self.context, self.buffer, index_buffer, offset, index_type)
    }

    /// Binds vertex buffers.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindVertexBuffers.html
    pub fn cmd_bind_vertex_buffer(
        &self,
        first_binding: u32,
        vertex_buffers: &[vk::Buffer],
        offsets: &[u64],
    ) {
        cmd_bind_vertex_buffer(
            &self.context,
            self.buffer,
            first_binding,
            vertex_buffers,
            offsets,
        )
    }

    /// Draws primitives.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDraw.html
    pub fn cmd_draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.context.logical_device.cmd_draw(
                self.buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        };
    }
}

#[inline]
fn begin(context: &Context, buffer: vk::CommandBuffer) -> Result<()> {
    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { context.logical_device.begin_command_buffer(buffer, &info)? };

    Ok(())
}

#[inline]
fn end(context: &Context, buffer: vk::CommandBuffer) -> Result<()> {
    unsafe { context.logical_device.end_command_buffer(buffer)? };

    Ok(())
}

#[inline]
fn cmd_bind_index_buffer(
    context: &Context,
    buffer: vk::CommandBuffer,
    index_buffer: vk::Buffer,
    offset: u64,
    index_type: vk::IndexType,
) {
    unsafe {
        context
            .logical_device
            .cmd_bind_index_buffer(buffer, index_buffer, offset, index_type)
    };
}

#[inline]
fn cmd_bind_vertex_buffer(
    context: &Context,
    buffer: vk::CommandBuffer,
    first_binding: u32,
    vertex_buffers: &[vk::Buffer],
    offsets: &[u64],
) {
    unsafe {
        context.logical_device.cmd_bind_vertex_buffers(
            buffer,
            first_binding,
            vertex_buffers,
            offsets,
        )
    };
}

#[inline]
fn cmd_bind_pipeline(
    context: &Context,
    buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    unsafe {
        context
            .logical_device
            .cmd_bind_pipeline(buffer, pipeline_bind_point, pipeline)
    };
}

#[inline]
fn cmd_copy_buffer(
    context: &Context,
    buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
) {
    let regions = [vk::BufferCopy::builder()
        .dst_offset(dst_offset)
        .src_offset(src_offset)
        .size(size)
        .build()];
    unsafe {
        context
            .logical_device
            .cmd_copy_buffer(buffer, src_buffer, dst_buffer, &regions)
    };
}
