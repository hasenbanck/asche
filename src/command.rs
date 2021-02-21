//! Implements command pools and command buffers.

use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::{Handle, Offset2D};

use crate::context::{Context, TagName};
use crate::{QueueType, Result};

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
    pub fn create_command_buffer(&mut self) -> Result<CommandBuffer> {
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

        let command_buffer = CommandBuffer::new(self.context.clone(), command_buffers[0]);

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
    pub(crate) encoder: CommandEncoder,
}

impl CommandBuffer {
    pub(crate) fn new(context: Arc<Context>, buffer: vk::CommandBuffer) -> Self {
        Self {
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

    /// Resets a command buffer.
    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.encoder
                .context
                .logical_device
                .reset_command_buffer(self.encoder.buffer, vk::CommandBufferResetFlags::empty())?
        };

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

    /// Sets the viewport.
    pub fn set_viewport(&self, viewport: vk::Viewport) {
        unsafe {
            self.context
                .logical_device
                .cmd_set_viewport(self.buffer, 0, &[viewport])
        };
    }

    /// Sets the scissor rectangle.
    pub fn set_scissor(&self, scissor_rect: vk::Rect2D) {
        unsafe {
            self.context
                .logical_device
                .cmd_set_scissor(self.buffer, 0, &[scissor_rect])
        };
    }

    /// Clears an attachment.
    pub fn clear_attachment(
        &self,
        attachment_index: u32,
        clear_rect: vk::Rect2D,
        clear_value: vk::ClearValue,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        let clear_attachment = vk::ClearAttachment::builder()
            .color_attachment(attachment_index)
            .clear_value(clear_value)
            .aspect_mask(aspect_mask);
        let rect = vk::ClearRect {
            rect: clear_rect,
            base_array_layer: 0,
            layer_count: 0,
        };
        unsafe {
            self.context.logical_device.cmd_clear_attachments(
                self.buffer,
                &[clear_attachment.build()],
                &[rect],
            );
        };
    }
}
