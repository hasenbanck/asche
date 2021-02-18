use ash::version::DeviceV1_0;
use ash::vk;

use crate::device::Queue;
use crate::{Device, Result};

/// Abstracts a Vulkan swapchain.
pub struct Swapchain {
    loader: ash::extensions::khr::Swapchain,
    raw: vk::SwapchainKHR,
    image_views: Vec<vk::ImageView>,
    present_complete_semaphore: vk::Semaphore,
}

/// Configures a swapchain
pub(crate) struct SwapchainDescriptor {
    pub(crate) graphic_queue_family_index: u32,
    pub(crate) extend: vk::Extent2D,
    pub(crate) pre_transform: vk::SurfaceTransformFlagsKHR,
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
            Some(mut osc) => {
                Self::destroy_resources(
                    &device.raw,
                    &mut osc.image_views,
                    &osc.present_complete_semaphore,
                );
                osc.raw
            }
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
            .pre_transform(descriptor.pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(descriptor.presentation_mode)
            .old_swapchain(old_swapchain)
            .clipped(true);

        let swapchain_loader =
            ash::extensions::khr::Swapchain::new(&device.context.instance, &device.raw);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        if old_swapchain != vk::SwapchainKHR::null() {
            unsafe {
                swapchain_loader.destroy_swapchain(old_swapchain, None);
            };
        }

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let image_views =
            Swapchain::create_image_views(device, &images, descriptor.format, images.len())?;

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let present_complete_semaphore =
            unsafe { device.raw.create_semaphore(&semaphore_create_info, None)? };

        Ok(Self {
            loader: swapchain_loader,
            raw: swapchain,
            image_views,
            present_complete_semaphore,
        })
    }

    /// Acquires the next frame that can be rendered into to being presented. Will block when no image in the swapchain is available.
    pub(crate) fn acquire_next_frame(&self) -> Result<SwapchainFrame> {
        let (index, _) = unsafe {
            self.loader.acquire_next_image(
                self.raw,
                std::u64::MAX,
                self.present_complete_semaphore,
                vk::Fence::null(),
            )?
        };
        Ok(SwapchainFrame {
            view: self.image_views[index as usize],
            index,
        })
    }

    /// Queues the given frame into the graphic queue.
    pub(crate) fn queue_frame(&self, frame: SwapchainFrame, graphic_queue: &Queue) -> Result<()> {
        let wait_semaphors = [self.present_complete_semaphore];
        let swapchains = [self.raw];
        let image_indices = [frame.index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.loader
                .queue_present(graphic_queue.raw, &present_info)?
        };

        Ok(())
    }

    pub(crate) fn destroy(&mut self, logical_device: &ash::Device) {
        Self::destroy_resources(
            logical_device,
            &mut self.image_views,
            &self.present_complete_semaphore,
        );
        unsafe { self.loader.destroy_swapchain(self.raw, None) };
    }

    fn destroy_resources(
        logical_device: &ash::Device,
        image_views: &mut [vk::ImageView],
        present_complete_semaphore: &vk::Semaphore,
    ) {
        unsafe {
            logical_device.destroy_semaphore(*present_complete_semaphore, None);
            for image_view in image_views {
                logical_device.destroy_image_view(*image_view, None);
            }
        };
    }

    fn create_image_views(
        device: &Device,
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
            let imageview = unsafe { device.raw.create_image_view(&imageview_create_info, None) }?;
            image_views.push(imageview);
        }

        Ok(image_views)
    }
}

/// Swapchain frame.
pub struct SwapchainFrame {
    pub view: vk::ImageView,
    pub index: u32,
}
