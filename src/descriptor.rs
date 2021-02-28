use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::Handle;

use crate::{Context, DescriptorSetUpdate, Result, UpdateDescriptorSetDescriptor};

/// Wraps a descriptor pool.
pub struct DescriptorPool {
    context: Arc<Context>,
    /// The raw Vulkan descriptor pool.
    pub raw: vk::DescriptorPool,
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_descriptor_pool(self.raw, None);
        };
    }
}

impl DescriptorPool {
    pub(crate) fn new(context: Arc<Context>, pool: vk::DescriptorPool) -> Self {
        Self { context, raw: pool }
    }

    /// Creates a new descriptor set with the given `DescriptorSetLayout``.
    pub fn create_descriptor_set(
        &self,
        name: &str,
        layout: &DescriptorSetLayout,
    ) -> Result<DescriptorSet> {
        let layouts = [layout.raw];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.raw)
            .set_layouts(&layouts);
        let set = unsafe {
            self.context
                .logical_device
                .allocate_descriptor_sets(&info)?[0]
        };

        self.context
            .set_object_name(name, vk::ObjectType::DESCRIPTOR_SET, set.as_raw())?;

        Ok(DescriptorSet {
            context: self.context.clone(),
            pool: self.raw,
            raw: set,
        })
    }

    /// Frees all descriptor sets allocated from the pool. Invalidates all descriptor sets from the pool.
    pub fn free_sets(&self) -> Result<()> {
        unsafe {
            self.context
                .logical_device
                .reset_descriptor_pool(self.raw, vk::DescriptorPoolResetFlags::empty())?;
        };
        Ok(())
    }
}

/// Wraps a descriptor set layout.
pub struct DescriptorSetLayout {
    context: Arc<Context>,
    /// The raw Vulkan descriptor set layout.
    pub raw: vk::DescriptorSetLayout,
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device
                .destroy_descriptor_set_layout(self.raw, None);
        };
    }
}

impl DescriptorSetLayout {
    pub(crate) fn new(context: Arc<Context>, layout: vk::DescriptorSetLayout) -> Self {
        Self {
            context,
            raw: layout,
        }
    }
}

/// Wraps a descriptor set.
pub struct DescriptorSet {
    context: Arc<Context>,
    pool: vk::DescriptorPool,
    /// The raw Vulkan descriptor pool.
    pub raw: vk::DescriptorSet,
}

impl DescriptorSet {
    /// Creates a new descriptor set with the given `DescriptorSetLayout``.
    pub fn update(&self, descriptor: &UpdateDescriptorSetDescriptor) {
        let write = vk::WriteDescriptorSet::builder().dst_set(self.raw);

        match descriptor.update {
            DescriptorSetUpdate::Sampler { .. } => {
                unimplemented!("Not implemented yet.")
            }
            DescriptorSetUpdate::CombinedImageSampler {
                sampler,
                image_view,
                image_layout,
            } => {
                let image_info = [vk::DescriptorImageInfo {
                    sampler: sampler.raw,
                    image_view: image_view.raw,
                    image_layout,
                }];
                self.inner_update(
                    write
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info),
                );
            }
            DescriptorSetUpdate::SampledImage { .. } => {
                unimplemented!("Not implemented yet.")
            }
            DescriptorSetUpdate::StorageImage { .. } => {
                unimplemented!("Not implemented yet.")
            }
            DescriptorSetUpdate::UniformTexelBuffer { .. } => {
                unimplemented!("Not implemented yet.")
            }
            DescriptorSetUpdate::StorageTexelBuffer { .. } => {
                unimplemented!("Not implemented yet.")
            }
            DescriptorSetUpdate::UniformBuffer {
                buffer,
                offset,
                range,
            } => {
                let buffer_info = [vk::DescriptorBufferInfo {
                    buffer: buffer.raw,
                    offset,
                    range,
                }];
                self.inner_update(
                    write
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buffer_info),
                );
            }
            DescriptorSetUpdate::StorageBuffer {
                buffer,
                offset,
                range,
            } => {
                let buffer_info = [vk::DescriptorBufferInfo {
                    buffer: buffer.raw,
                    offset,
                    range,
                }];
                self.inner_update(
                    write
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buffer_info),
                );
            }
            DescriptorSetUpdate::UniformBufferDynamic {
                buffer,
                offset,
                range,
            } => {
                let buffer_info = [vk::DescriptorBufferInfo {
                    buffer: buffer.raw,
                    offset,
                    range,
                }];
                self.inner_update(
                    write
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .buffer_info(&buffer_info),
                );
            }
            DescriptorSetUpdate::StorageBufferDynamic {
                buffer,
                offset,
                range,
            } => {
                let buffer_info = [vk::DescriptorBufferInfo {
                    buffer: buffer.raw,
                    offset,
                    range,
                }];
                self.inner_update(
                    write
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                        .buffer_info(&buffer_info),
                );
            }
            DescriptorSetUpdate::InputAttachment { .. } => {
                unimplemented!("Not implemented yet.")
            }
        }
    }

    fn inner_update(&self, builder: vk::WriteDescriptorSetBuilder) {
        let writes = &[builder.build()];
        unsafe {
            self.context
                .logical_device
                .update_descriptor_sets(writes, &[]);
        };
    }

    /// Resets the descriptor set. Needs to be created from a pool with the
    /// `vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET` flag set.
    pub fn reset(&mut self) {
        let sets = [self.raw];
        unsafe {
            self.context
                .logical_device
                .free_descriptor_sets(self.pool, &sets);
        };
    }
}
