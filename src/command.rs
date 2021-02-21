//! Implements command pools and command buffers.

use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::Handle;

use crate::context::{Context, TagName};
use crate::{QueueType, RenderPass, Result};

/// A wrapped command pool.
pub struct CommandPool {
    context: Arc<Context>,
    raw: vk::CommandPool,
    queue_type: QueueType,
    id: u64,
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
        raw: vk::CommandPool,
        queue_type: QueueType,
        id: u64,
    ) -> Result<Self> {
        context.set_object_name(
            &format!("command pool {}", id),
            vk::ObjectType::COMMAND_POOL,
            raw.as_raw(),
        )?;

        let bytes: [u8; 1] = unsafe { std::mem::transmute(queue_type as u8) };
        context.set_object_tag(
            TagName::QueueType,
            &bytes,
            vk::ObjectType::COMMAND_POOL,
            raw.as_raw(),
        )?;

        Ok(Self {
            context,
            raw,
            queue_type,
            id,
            command_buffer_counter: 0,
        })
    }

    /// Creates a new command buffer.
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
            &format!("command buffer {}", self.command_buffer_counter,),
            vk::ObjectType::COMMAND_BUFFER,
            command_buffers[0].as_raw(),
        )?;

        let bytes: [u8; 8] = unsafe { std::mem::transmute(self.id) };
        self.context.set_object_tag(
            TagName::BufferPoolIndex,
            &bytes,
            vk::ObjectType::COMMAND_BUFFER,
            command_buffers[0].as_raw(),
        )?;

        let bytes: [u8; 1] = unsafe { std::mem::transmute(self.queue_type as u8) };
        self.context.set_object_tag(
            TagName::QueueType,
            &bytes,
            vk::ObjectType::COMMAND_BUFFER,
            command_buffers[0].as_raw(),
        )?;

        let command_buffer = CommandBuffer::new(
            self.context.clone(),
            command_buffers[0],
            timeline_wait_value,
            timeline_signal_value,
        );

        self.command_buffer_counter += 1;

        Ok(command_buffer)
    }

    /// Resets a command pool.
    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.context
                .logical_device
                .reset_command_pool(self.raw, vk::CommandPoolResetFlags::empty())?
        };

        Ok(())
    }
}

/// A wrapped command buffer.
pub struct CommandBuffer {
    pub(crate) timeline_wait_value: u64,
    pub(crate) timeline_signal_value: u64,
    pub(crate) encoder: CommandEncoder,
}

impl CommandBuffer {
    pub(crate) fn new(
        context: Arc<Context>,
        buffer: vk::CommandBuffer,
        timeline_wait_value: u64,
        timeline_signal_value: u64,
    ) -> Self {
        Self {
            timeline_wait_value,
            timeline_signal_value,
            encoder: CommandEncoder { context, buffer },
        }
    }

    /// Records the command buffer actions with the help of an encoder.
    pub fn record<F>(&self, exec: F) -> Result<()>
    where
        F: Fn(&CommandEncoder) -> Result<()>,
    {
        self.encoder.begin()?;
        exec(&self.encoder)?;
        self.encoder.end()?;

        Ok(())
    }
}

/// Used to encode command for a command buffer.
pub struct CommandEncoder {
    pub(crate) context: Arc<Context>,
    pub(crate) buffer: vk::CommandBuffer,
}

impl CommandEncoder {
    /// Begins a command buffer.
    fn begin(&self) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.context
                .logical_device
                .begin_command_buffer(self.buffer, &info)?
        };

        Ok(())
    }

    /// Ends a command buffer.
    fn end(&self) -> Result<()> {
        unsafe {
            self.context
                .logical_device
                .end_command_buffer(self.buffer)?
        };

        Ok(())
    }

    /// Returns a renderpass encoder. Drop once finished recording.
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
}

/// Used to encode renderpass commands of a command buffer.
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
}
