//! Implements the internal context.
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hasher;

use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::Handle;
use parking_lot::Mutex;

use crate::{
    Instance, RenderPass, RenderPassColorAttachmentDescriptor, RenderPassDepthAttachmentDescriptor,
    Result,
};

/// The internal context.
pub struct Context {
    /// The wrapped Vulkan instance.
    pub instance: Instance,
    /// The raw logical Vulkan device.
    pub logical_device: ash::Device,
    /// The raw physical Vulkan device.
    pub physical_device: vk::PhysicalDevice,
    /// The memory allocator.
    pub allocator: Mutex<vk_alloc::Allocator>,
    /// The framebuffer.
    framebuffers: Mutex<HashMap<u64, vk::Framebuffer>>,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.device_wait_idle().unwrap();
            self.destroy_framebuffer();
            self.allocator.lock().free_all();
            self.logical_device.destroy_device(None);
        };
    }
}

impl Context {
    /// Creates a new context.
    pub(crate) fn new(
        instance: Instance,
        logical_device: ash::Device,
        physical_device: vk::PhysicalDevice,
        allocator: vk_alloc::Allocator,
    ) -> Self {
        Self {
            instance,
            logical_device,
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
        hasher.write_u64(render_pass.raw.as_raw());
        for color_attachment in color_attachments {
            hasher.write_u64(color_attachment.attachment.as_raw());
        }
        if let Some(depth_attachment) = depth_attachment {
            hasher.write_u64(depth_attachment.attachment.as_raw());
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

        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.raw)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        let framebuffer = unsafe {
            self.logical_device
                .create_framebuffer(&framebuffer_info, None)?
        };

        Ok(framebuffer)
    }

    pub(crate) fn destroy_framebuffer(&self) {
        for (_, framebuffer) in self.framebuffers.lock().drain() {
            unsafe {
                self.logical_device.destroy_framebuffer(framebuffer, None);
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
        let info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_name(&name)
            .object_type(object_type)
            .object_handle(object_handle);
        unsafe {
            self.instance
                .debug_utils
                .debug_utils_set_object_name(self.logical_device.handle(), &info)?
        };

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
