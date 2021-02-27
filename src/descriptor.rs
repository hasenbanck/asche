use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::Handle;

use crate::{Context, Result};

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
    ) -> Result<vk::DescriptorSet> {
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

        Ok(set)
    }

    /// Creates a new descriptor set with the given `DescriptorSetLayout``.
    pub fn update_descriptor_set(
        &self,
        writes: &[vk::WriteDescriptorSet],
        copies: &[vk::CopyDescriptorSet],
    ) {
        unsafe {
            self.context
                .logical_device
                .update_descriptor_sets(writes, copies);
        };
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
