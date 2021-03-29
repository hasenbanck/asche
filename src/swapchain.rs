use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "smallvec")]
use smallvec::SmallVec;
#[cfg(feature = "tracing")]
use tracing::{error, info};

use crate::context::Context;
use crate::{
    AscheError, GraphicsQueue, ImageView, RenderPass, RenderPassColorAttachmentDescriptor,
    RenderPassDepthAttachmentDescriptor, Result,
};

/// Swapchain frame.
#[derive(Debug)]
pub struct SwapchainFrame {
    /// The index of the swapchain.
    pub index: u32,
    /// The Vulkan image view.
    pub view: vk::ImageView,
}

/// Abstracts a Vulkan swapchain.
#[derive(Debug)]
pub struct Swapchain {
    /// The framebuffer.
    framebuffers: HashMap<u64, vk::Framebuffer>,
    graphic_queue_family_index: u32,
    presentation_mode: vk::PresentModeKHR,
    size: Option<u32>,
    swapchain: Option<SwapchainInner>,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    context: Arc<Context>,
}

impl Swapchain {
    pub(crate) fn new(
        context: Arc<Context>,
        graphic_queue_family_index: u32,
        presentation_mode: vk::PresentModeKHR,
        format: vk::Format,
        color_space: vk::ColorSpaceKHR,
    ) -> Result<Self> {
        let mut swapchain = Self {
            framebuffers: HashMap::with_capacity(3),
            graphic_queue_family_index,
            presentation_mode,
            size: None,
            swapchain: None,
            format,
            color_space,
            context,
        };

        swapchain.recreate(None)?;

        Ok(swapchain)
    }

    /// Recreates the swapchain. Needs to be called if the surface has changed.
    pub fn recreate(&mut self, window_extend: Option<vk::Extent2D>) -> Result<()> {
        self.destroy_framebuffer();

        #[cfg(feature = "tracing")]
        info!(
            "Creating swapchain with format {:?} and color space {:?}",
            self.format, self.color_space
        );

        let formats = self.query_formats()?;

        let capabilities = self.query_surface_capabilities()?;
        let presentation_mode = self.query_presentation_mode()?;

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let extent = match capabilities.current_extent.width {
            u32::MAX => window_extend.unwrap_or_default(),
            _ => capabilities.current_extent,
        };

        let pre_transform = if capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY_KHR)
        {
            vk::SurfaceTransformFlagBitsKHR::IDENTITY_KHR
        } else {
            capabilities.current_transform
        };

        let format = formats
            .iter()
            .find(|f| f.format == self.format && f.color_space == self.color_space)
            .ok_or(AscheError::SwapchainFormatIncompatible)?;

        let presentation_mode = *presentation_mode
            .iter()
            .find(|m| **m == self.presentation_mode)
            .ok_or(AscheError::PresentationModeUnsupported)?;

        let old_swapchain = self.swapchain.take();

        let swapchain = SwapchainInner::new(
            self.context.clone(),
            SwapchainDescriptor {
                graphic_queue_family_index: self.graphic_queue_family_index,
                extent,
                pre_transform,
                format: format.format,
                color_space: format.color_space,
                presentation_mode,
                image_count,
            },
            old_swapchain,
        )?;

        #[cfg(feature = "tracing")]
        info!("Swapchain has {} image(s)", image_count);

        self.swapchain.replace(swapchain);

        Ok(())
    }

    fn query_formats(&self) -> Result<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            self.context
                .instance
                .raw
                .get_physical_device_surface_formats_khr(
                    self.context.physical_device,
                    self.context.instance.surface,
                    None,
                )
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to get the physical device surface formats: {}", err);
            AscheError::VkResult(err)
        })
    }

    fn query_surface_capabilities(&self) -> Result<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            self.context
                .instance
                .raw
                .get_physical_device_surface_capabilities_khr(
                    self.context.physical_device,
                    self.context.instance.surface,
                    None,
                )
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!(
                "Unable to get the physical device surface capabilities: {}",
                err
            );
            AscheError::VkResult(err)
        })
    }

    fn query_presentation_mode(&self) -> Result<Vec<vk::PresentModeKHR>> {
        unsafe {
            self.context
                .instance
                .raw
                .get_physical_device_surface_present_modes_khr(
                    self.context.physical_device,
                    self.context.instance.surface,
                    None,
                )
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to get the physical device surface modes: {}", err);
            AscheError::VkResult(err)
        })
    }

    /// Returns the frame count of the swapchain.
    pub fn frame_count(&self) -> Result<u32> {
        let capabilities = self.query_surface_capabilities()?;

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        Ok(image_count)
    }

    /// Gets the next frame the program can render into.
    pub fn next_frame(&self) -> Result<SwapchainFrame> {
        self.swapchain
            .as_ref()
            .ok_or(AscheError::SwapchainNotInitialized)?
            .get_next_frame()
    }

    /// Queues the frame in the presentation queue.
    pub fn queue_frame(&self, graphics_queue: &GraphicsQueue, frame: SwapchainFrame) -> Result<()> {
        self.swapchain
            .as_ref()
            .ok_or(AscheError::SwapchainNotInitialized)?
            .queue_frame(frame, graphics_queue.raw)
    }

    /// Re-uses a cached framebuffer or creates a new one.
    pub(crate) fn next_framebuffer(
        &mut self,
        render_pass: &RenderPass,
        color_attachments: &[RenderPassColorAttachmentDescriptor],
        depth_attachment: &Option<RenderPassDepthAttachmentDescriptor>,
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
        let framebuffer = if let Some(framebuffer) = self.framebuffers.get(&hash) {
            *framebuffer
        } else {
            created = true;
            self.create_framebuffer(render_pass, color_attachments, depth_attachment, extent)?
        };

        if created {
            self.framebuffers.insert(hash, framebuffer);
        }

        Ok(framebuffer)
    }

    fn create_framebuffer(
        &self,
        render_pass: &RenderPass,
        color_attachments: &[RenderPassColorAttachmentDescriptor],
        depth_attachment: &Option<RenderPassDepthAttachmentDescriptor>,
        extent: vk::Extent2D,
    ) -> Result<vk::Framebuffer> {
        let attachments = color_attachments
            .iter()
            .map(|x| x.attachment)
            .chain(depth_attachment.iter().map(|x| x.attachment));

        #[cfg(feature = "smallvec")]
        let attachments = attachments.collect::<SmallVec<[vk::ImageView; 4]>>();

        #[cfg(not(feature = "smallvec"))]
        let attachments = attachments.collect::<Vec<vk::ImageView>>();

        let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
            .render_pass(render_pass.raw)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let framebuffer = unsafe {
            self.context
                .device
                .create_framebuffer(&framebuffer_info, None, None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create a frame buffer: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(framebuffer)
    }

    fn destroy_framebuffer(&mut self) {
        for (_, framebuffer) in self.framebuffers.drain() {
            unsafe {
                self.context
                    .device
                    .destroy_framebuffer(Some(framebuffer), None);
            }
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.destroy_framebuffer();
    }
}

/// The inner abstraciton of the swapchain.
#[derive(Debug)]
pub struct SwapchainInner {
    present_complete_semaphore: vk::Semaphore,
    image_views: Vec<ImageView>,
    raw: vk::SwapchainKHR,
    context: Arc<Context>,
}

/// Configures a swapchain
#[derive(Clone, Debug)]
struct SwapchainDescriptor {
    graphic_queue_family_index: u32,
    extent: vk::Extent2D,
    pre_transform: vk::SurfaceTransformFlagBitsKHR,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    presentation_mode: vk::PresentModeKHR,
    image_count: u32,
}

impl SwapchainInner {
    /// Creates a new `Swapchain`.
    fn new(
        context: Arc<Context>,
        descriptor: SwapchainDescriptor,
        old_swapchain: Option<SwapchainInner>,
    ) -> Result<Self> {
        let old_swapchain = match old_swapchain {
            Some(mut osc) => {
                let swapchain = osc.raw;

                // We need to destroy the associated resources of the swapchain, before we can
                // try to reuse the vk::SwapchainKHR when creating the new swapchain.
                Self::destroy_resources(&context.device, &mut osc.present_complete_semaphore);
                // We set the raw handler to null, so that drop doesn't try to destroy it again,
                // since "create_swapchain()" will do that for us.
                osc.raw = vk::SwapchainKHR::null();

                swapchain
            }
            None => vk::SwapchainKHR::null(),
        };

        let graphic_family_index = &[descriptor.graphic_queue_family_index];
        let swapchain_create_info = vk::SwapchainCreateInfoKHRBuilder::new()
            .surface(context.instance.surface)
            .min_image_count(descriptor.image_count)
            .image_format(descriptor.format)
            .image_color_space(descriptor.color_space)
            .image_extent(descriptor.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(graphic_family_index)
            .pre_transform(descriptor.pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
            .present_mode(descriptor.presentation_mode)
            .old_swapchain(old_swapchain)
            .clipped(true);

        let swapchain = unsafe {
            context
                .device
                .create_swapchain_khr(&swapchain_create_info, None, None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create a swapchain: {}", err);
            AscheError::VkResult(err)
        })?;

        let images =
            unsafe { context.device.get_swapchain_images_khr(swapchain, None) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to get the swapchain images: {}", err);
                AscheError::VkResult(err)
            })?;

        let image_views =
            SwapchainInner::create_image_views(&context, &images, descriptor.format, images.len())?;

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let present_complete_semaphore = unsafe {
            context
                .device
                .create_semaphore(&semaphore_create_info, None, None)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to create the presentation semaphore: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(Self {
            context,
            raw: swapchain,
            image_views,
            present_complete_semaphore,
        })
    }

    /// Acquires the next frame that can be rendered into to being presented. Will block when no image in the swapchain is available.
    fn get_next_frame(&self) -> Result<SwapchainFrame> {
        let info = vk::AcquireNextImageInfoKHRBuilder::new()
            .semaphore(self.present_complete_semaphore)
            .device_mask(1)
            .swapchain(self.raw)
            .timeout(u64::MAX);

        let index =
            unsafe { self.context.device.acquire_next_image2_khr(&info, None) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to acquire the next frame image: {}", err);
                AscheError::VkResult(err)
            })?;
        let view = self.image_views[index as usize].raw;
        Ok(SwapchainFrame { index, view })
    }

    /// Queues the given frame into the graphic queue.
    fn queue_frame(&self, frame: SwapchainFrame, graphic_queue: vk::Queue) -> Result<()> {
        let wait_semaphors = [self.present_complete_semaphore];
        let swapchains = [self.raw];
        let image_indices = [frame.index];
        let present_info = vk::PresentInfoKHRBuilder::new()
            .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.context
                .device
                .queue_present_khr(graphic_queue, &present_info)
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to queue the next frame: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(())
    }

    fn create_image_views(
        context: &Arc<Context>,
        images: &[vk::Image],
        format: vk::Format,
        size: usize,
    ) -> Result<Vec<ImageView>> {
        let mut image_views = Vec::with_capacity(size);

        for image in images.iter() {
            let imageview_create_info = vk::ImageViewCreateInfoBuilder::new()
                .view_type(vk::ImageViewType::_2D)
                .format(format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(*image);
            let raw = unsafe {
                context
                    .device
                    .create_image_view(&imageview_create_info, None, None)
            }
            .map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to create a swapchain image view: {}", err);
                AscheError::VkResult(err)
            })?;

            image_views.push(ImageView::new(raw, context.clone()));
        }

        Ok(image_views)
    }

    fn destroy_resources(
        device: &erupt::DeviceLoader,
        present_complete_semaphore: &mut vk::Semaphore,
    ) {
        unsafe {
            device.destroy_semaphore(Some(*present_complete_semaphore), None);
            *present_complete_semaphore = vk::Semaphore::null();
        };
    }
}

impl Drop for SwapchainInner {
    fn drop(&mut self) {
        Self::destroy_resources(&self.context.device, &mut self.present_complete_semaphore);
        unsafe {
            self.context
                .device
                .destroy_swapchain_khr(Some(self.raw), None)
        };
    }
}
