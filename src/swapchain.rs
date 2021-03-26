use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing::error;

use crate::context::Context;
use crate::{AscheError, ImageView, Result};

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
    present_complete_semaphore: vk::Semaphore,
    image_views: Vec<ImageView>,
    raw: vk::SwapchainKHR,
    context: Arc<Context>,
}

/// Configures a swapchain
#[derive(Clone, Debug)]
pub(crate) struct SwapchainDescriptor {
    pub(crate) graphic_queue_family_index: u32,
    pub(crate) extent: vk::Extent2D,
    pub(crate) pre_transform: vk::SurfaceTransformFlagBitsKHR,
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
            Swapchain::create_image_views(&context, &images, descriptor.format, images.len())?;

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
    pub(crate) fn get_next_frame(&self) -> Result<SwapchainFrame> {
        let info = vk::AcquireNextImageInfoKHRBuilder::new()
            .semaphore(self.present_complete_semaphore)
            .device_mask(1)
            .swapchain(self.raw)
            .timeout(std::u64::MAX);

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
    pub(crate) fn queue_frame(
        &self,
        frame: SwapchainFrame,
        graphic_queue: vk::Queue,
    ) -> Result<()> {
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

impl Drop for Swapchain {
    fn drop(&mut self) {
        Self::destroy_resources(&self.context.device, &mut self.present_complete_semaphore);
        unsafe {
            self.context
                .device
                .destroy_swapchain_khr(Some(self.raw), None)
        };
    }
}
