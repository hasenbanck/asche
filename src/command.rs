//! Implements command pools and command buffers.

use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::Handle;

use crate::context::Context;
use crate::Result;

/// A wrapped command pool.
pub struct CommandPool {
    pub(crate) id: u32,
    pub(crate) context: Arc<Context>,
    pub(crate) raw: vk::CommandPool,
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
    /// Creates a new command buffer.
    pub fn create_command_buffer(&mut self) -> Result<CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.raw)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            self.context
                .logical_device
                .allocate_command_buffers(&info)?
        };

        self.context.set_object_name(
            &format!(
                "command pool {} command buffer {}",
                self.id,
                command_buffer[0].as_raw(),
            ),
            vk::ObjectType::COMMAND_BUFFER,
            command_buffer[0].as_raw(),
        )?;

        Ok(CommandBuffer {
            context: self.context.clone(),
            raw: command_buffer[0],
        })
    }

    /// Creates new command buffers.
    pub fn create_command_buffers(&mut self, count: u32) -> Result<Vec<CommandBuffer>> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.raw)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        let command_buffers = unsafe {
            self.context
                .logical_device
                .allocate_command_buffers(&info)?
        };

        let command_buffers = command_buffers
            .iter()
            .map(|buffer| CommandBuffer {
                context: self.context.clone(),
                raw: *buffer,
            })
            .collect();

        Ok(command_buffers)
    }
}

/// A wrapped command buffer.
pub struct CommandBuffer {
    pub(crate) context: Arc<Context>,
    pub(crate) raw: vk::CommandBuffer,
}

impl CommandBuffer {
    /// Sets the viewport.
    pub fn set_viewport(&self, viewport: vk::Viewport) {
        unsafe {
            self.context
                .logical_device
                .cmd_set_viewport(self.raw, 0, &[viewport]);
        };
    }

    /// Sets the scissor rectangle.
    pub fn set_scissor(&self, scissor_rect: vk::Rect2D) {
        unsafe {
            self.context
                .logical_device
                .cmd_set_scissor(self.raw, 0, &[scissor_rect]);
        };
    }
}
