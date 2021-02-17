use ash::version::DeviceV1_0;
use ash::vk;

use crate::{Device, Result};

/// Abstracts a Vulkan swapchain.
pub struct Swapchain {
    loader: ash::extensions::khr::Swapchain,
    raw: vk::SwapchainKHR,
    image_views: Vec<vk::ImageView>,
}

/// Configures a swapchain
pub(crate) struct SwapchainDescriptor {
    pub(crate) graphic_queue_family_index: u32,
    pub(crate) extend: vk::Extent2D,
    pub(crate) transform: vk::SurfaceTransformFlagsKHR,
    pub(crate) format: vk::Format,
    pub(crate) color_space: vk::ColorSpaceKHR,
    pub(crate) presentation_mode: vk::PresentModeKHR,
    pub(crate) image_count: u32,
}

impl Swapchain {
    /// Creates a new `Swapchain`.
    pub(crate) fn new(
        device: &Device,
        descriptor: SwapchainDescriptor,
        old_swapchain: Option<Swapchain>,
    ) -> Result<Self> {
        let old_swapchain = match old_swapchain {
            Some(osc) => osc.raw,
            None => vk::SwapchainKHR::null(),
        };

        let graphic_family_index = &[descriptor.graphic_queue_family_index];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(device.context.surface)
            .min_image_count(descriptor.image_count)
            .image_format(descriptor.format)
            .image_color_space(descriptor.color_space)
            .image_extent(descriptor.extend)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(graphic_family_index)
            .pre_transform(descriptor.transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(descriptor.presentation_mode)
            .old_swapchain(old_swapchain);

        let swapchain_loader =
            ash::extensions::khr::Swapchain::new(&device.context.instance, &device.raw);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let mut image_views = Vec::with_capacity(images.len());

        for image in &images {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let imageview_create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(descriptor.format)
                .subresource_range(*subresource_range);
            let imageview = unsafe { device.raw.create_image_view(&imageview_create_info, None) }?;
            image_views.push(imageview);
        }

        Ok(Self {
            loader: swapchain_loader,
            raw: swapchain,
            image_views,
        })
    }

    pub(crate) fn destroy(&mut self, logical_device: &ash::Device) {
        unsafe {
            for image_view in &self.image_views {
                logical_device.destroy_image_view(*image_view, None);
            }
            self.loader.destroy_swapchain(self.raw, None);
        };
    }
}
