//! Implements the internal context.
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hasher;

use erupt::vk;
use parking_lot::Mutex;
#[cfg(feature = "tracing")]
use tracing::error;

use crate::{
    AscheError, Instance, RenderPass, RenderPassColorAttachmentDescriptor,
    RenderPassDepthAttachmentDescriptor, Result,
};

/// The internal context.
pub struct Context {
    /// The framebuffer.
    framebuffers: Mutex<HashMap<u64, vk::Framebuffer>>,
    /// The memory allocator.
    pub allocator: Mutex<vk_alloc::Allocator>,
    /// The raw logical Vulkan device.
    pub device: erupt::DeviceLoader,
    /// The raw physical Vulkan device.
    pub physical_device: vk::PhysicalDevice,
    /// The wrapped Vulkan instance.
    pub instance: Instance,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.destroy_framebuffer();
            self.allocator.lock().cleanup(&self.device);
            self.device.destroy_device(None);
        };
    }
}

impl Context {
    /// Creates a new context.
    pub(crate) fn new(
        instance: Instance,
        device: erupt::DeviceLoader,
        physical_device: vk::PhysicalDevice,
        allocator: vk_alloc::Allocator,
    ) -> Self {
        Self {
            instance,
            device,
            physical_device,
            allocator: Mutex::new(allocator),
            framebuffers: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn get_framebuffer(
        &self,
        render_pass: &RenderPass,
        color_attachments: &[&RenderPassColorAttachmentDescriptor],
        depth_attachment: Option<&RenderPassDepthAttachmentDescriptor>,
        extent: vk::Extent2D,
    ) -> Result<vk::Framebuffer> {
        // Calculate the hash for the renderpass / attachment combination.
        let mut hasher = DefaultHasher::new();
        hasher.write_u64(render_pass.raw.0);
        for color_attachment in color_attachments {
            hasher.write_u64(color_attachment.attachment.0);
        }
        if let Some(depth_attachment) = depth_attachment {
            hasher.write_u64(depth_attachment.attachment.0);
        }
        let hash = hasher.finish();

        let mut created = false;
        let framebuffer = if let Some(framebuffer) = self.framebuffers.lock().get(&hash) {
            *framebuffer
        } else {
            created = true;
            self.create_framebuffer(render_pass, color_attachments, depth_attachment, extent)?
        };

        if created {
            self.framebuffers.lock().insert(hash, framebuffer);
        }

        Ok(framebuffer)
    }

    fn create_framebuffer(
        &self,
        render_pass: &RenderPass,
        color_attachments: &[&RenderPassColorAttachmentDescriptor],
        depth_attachment: Option<&RenderPassDepthAttachmentDescriptor>,
        extent: vk::Extent2D,
    ) -> Result<vk::Framebuffer> {
        let attachments: Vec<vk::ImageView> = color_attachments
            .iter()
            .map(|x| x.attachment)
            .chain(depth_attachment.iter().map(|x| x.attachment))
            .collect();

        let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
            .render_pass(render_pass.raw)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let framebuffer = unsafe {
            self.device
                .create_framebuffer(&framebuffer_info, None, None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create a frame buffer: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(framebuffer)
    }

    pub(crate) fn destroy_framebuffer(&self) {
        for (_, framebuffer) in self.framebuffers.lock().drain() {
            unsafe {
                self.device.destroy_framebuffer(Some(framebuffer), None);
            }
        }
    }

    /// Sets a debug name for an object.
    #[cfg(debug_assertions)]
    pub(crate) fn set_object_name(
        &self,
        name: &str,
        object_type: vk::ObjectType,
        object_handle: u64,
    ) -> Result<()> {
        let name = std::ffi::CString::new(name.to_owned())?;
        let info = vk::DebugUtilsObjectNameInfoEXTBuilder::new()
            .object_name(&name)
            .object_type(object_type)
            .object_handle(object_handle);
        unsafe { self.device.set_debug_utils_object_name_ext(&info) }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to set the debug object name: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(())
    }

    /// Sets a debug name for an object.
    #[cfg(not(debug_assertions))]
    pub(crate) fn set_object_name(
        &self,
        _name: &str,
        _object_type: vk::ObjectType,
        _object_handle: u64,
    ) -> Result<()> {
        Ok(())
    }
}
