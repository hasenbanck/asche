use std::collections::HashMap;
use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;

use crate::context::Context;
use crate::Result;

/// Swapchain frame.
pub struct SwapchainFrame {
    pub(crate) index: u32,
}

/// Abstracts a Vulkan swapchain.
pub struct Swapchain {
    context: Arc<Context>,
    loader: ash::extensions::khr::Swapchain,
    raw: vk::SwapchainKHR,
    image_views: Vec<vk::ImageView>,
    renderpass_framebuffers: HashMap<vk::RenderPass, Vec<vk::Framebuffer>>,
    present_complete_semaphore: vk::Semaphore,
    extent: vk::Extent2D,
}

/// Configures a swapchain
pub(crate) struct SwapchainDescriptor {
    pub(crate) graphic_queue_family_index: u32,
    pub(crate) extent: vk::Extent2D,
    pub(crate) pre_transform: vk::SurfaceTransformFlagsKHR,
    pub(crate) format: vk::Format,
    pub(crate) color_space: vk::ColorSpaceKHR,
    pub(crate) presentation_mode: vk::PresentModeKHR,
    pub(crate) image_count: u32,
}

impl Swapchain {
    /// Creates a new `Swapchain`.
    pub(crate) fn new(
        context: Arc<Context>,
        descriptor: SwapchainDescriptor,
        old_swapchain: Option<Swapchain>,
    ) -> Result<Self> {
        let (old_swapchain, renderpasses) = match old_swapchain {
            Some(mut osc) => {
                let swapchain = osc.raw;

                let renderpasses: Vec<vk::RenderPass> = osc
                    .renderpass_framebuffers
                    .iter()
                    .map(|(k, _)| *k)
                    .collect();

                // We need to destroy the associated resources of the swapchain, before we can
                // try to reuse the vk::SwapchainKHR when creating the new swapchain.
                Self::destroy_resources(
                    &context.logical_device,
                    &mut osc.image_views,
                    &mut osc.renderpass_framebuffers,
                    &mut osc.present_complete_semaphore,
                );
                // We set the raw handler to null, so that drop doesn't try to destroy it again,
                // since "create_swapchain()" will do that for us.
                osc.raw = vk::SwapchainKHR::null();

                (swapchain, renderpasses)
            }
            None => (vk::SwapchainKHR::null(), vec![]),
        };

        let graphic_family_index = &[descriptor.graphic_queue_family_index];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
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
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(descriptor.presentation_mode)
            .old_swapchain(old_swapchain)
            .clipped(true);

        let swapchain_loader =
            ash::extensions::khr::Swapchain::new(&context.instance.raw, &context.logical_device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let image_views =
            Swapchain::create_image_views(&context, &images, descriptor.format, images.len())?;

        let mut renderpass_framebuffers = HashMap::with_capacity(renderpasses.len());
        for renderpass in renderpasses {
            let framebuffers = Self::init_framebuffers(
                &context.logical_device,
                &image_views,
                renderpass,
                descriptor.extent,
            )?;
            renderpass_framebuffers.insert(renderpass, framebuffers);
        }

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let present_complete_semaphore = unsafe {
            context
                .logical_device
                .create_semaphore(&semaphore_create_info, None)?
        };

        Ok(Self {
            context,
            loader: swapchain_loader,
            raw: swapchain,
            image_views,
            renderpass_framebuffers,
            present_complete_semaphore,
            extent: descriptor.extent,
        })
    }

    /// Acquires the next frame that can be rendered into to being presented. Will block when no image in the swapchain is available.
    pub(crate) fn get_next_frame(&self) -> Result<SwapchainFrame> {
        let (index, _) = unsafe {
            self.loader.acquire_next_image(
                self.raw,
                std::u64::MAX,
                self.present_complete_semaphore,
                vk::Fence::null(),
            )?
        };
        Ok(SwapchainFrame { index })
    }

    /// Gets the framebuffer for a renderpass.
    pub(crate) fn get_frame_buffer(
        &self,
        render_pass: vk::RenderPass,
        index: u32,
    ) -> vk::Framebuffer {
        self.renderpass_framebuffers[&render_pass][index as usize]
    }

    /// Queues the given frame into the graphic queue.
    pub(crate) fn queue_frame(
        &self,
        frame: SwapchainFrame,
        graphic_queue: vk::Queue,
    ) -> Result<()> {
        let wait_semaphors = [self.present_complete_semaphore];
        let swapchains = [self.raw];
        let image_indices = [frame.index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { self.loader.queue_present(graphic_queue, &present_info)? };

        Ok(())
    }

    fn create_image_views(
        context: &Context,
        images: &[vk::Image],
        format: vk::Format,
        size: usize,
    ) -> Result<Vec<vk::ImageView>> {
        let mut image_views = Vec::with_capacity(size);

        for image in images.iter() {
            let imageview_create_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
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
            let imageview = unsafe {
                context
                    .logical_device
                    .create_image_view(&imageview_create_info, None)
            }?;
            image_views.push(imageview);
        }

        Ok(image_views)
    }

    /// Creates the framebuffers for a renderpass.
    pub(crate) fn create_renderpass_framebuffers(
        &mut self,
        renderpass: vk::RenderPass,
    ) -> Result<()> {
        let framebuffers = Self::init_framebuffers(
            &self.context.logical_device,
            &self.image_views,
            renderpass,
            self.extent,
        )?;
        self.renderpass_framebuffers
            .insert(renderpass, framebuffers);
        Ok(())
    }

    fn init_framebuffers(
        logical_device: &ash::Device,
        image_views: &[vk::ImageView],
        renderpass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>> {
        let mut framebuffers = Vec::with_capacity(image_views.len());
        for image_view in image_views {
            let view = [*image_view];
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass)
                .attachments(&view)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            let fb = unsafe { logical_device.create_framebuffer(&framebuffer_info, None)? };
            framebuffers.push(fb);
        }

        Ok(framebuffers)
    }

    fn destroy_resources(
        logical_device: &ash::Device,
        image_views: &mut [vk::ImageView],
        framebuffers: &mut HashMap<vk::RenderPass, Vec<vk::Framebuffer>>,
        present_complete_semaphore: &mut vk::Semaphore,
    ) {
        unsafe {
            logical_device.destroy_semaphore(*present_complete_semaphore, None);
            *present_complete_semaphore = vk::Semaphore::null();

            for (_, renderpass_framebuffers) in framebuffers {
                for framebuffer in renderpass_framebuffers {
                    logical_device.destroy_framebuffer(*framebuffer, None);
                    *framebuffer = vk::Framebuffer::null()
                }
            }

            for image_view in image_views {
                logical_device.destroy_image_view(*image_view, None);
                *image_view = vk::ImageView::null();
            }
        };
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        Self::destroy_resources(
            &self.context.logical_device,
            &mut self.image_views,
            &mut self.renderpass_framebuffers,
            &mut self.present_complete_semaphore,
        );
        unsafe { self.loader.destroy_swapchain(self.raw, None) };
    }
}
