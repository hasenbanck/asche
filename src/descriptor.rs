use std::sync::Arc;

use erupt::{vk, ExtendableFromConst};
#[cfg(feature = "tracing")]
use tracing1::error;

use crate::context::Context;
use crate::{AscheError, Result};

/// Wraps a descriptor pool.
#[derive(Debug)]
pub struct DescriptorPool {
    /// The raw Vulkan descriptor pool.
    pub raw: vk::DescriptorPool,
    context: Arc<Context>,
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_descriptor_pool(Some(self.raw), None);
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
        descriptor_count: Option<u32>,
    ) -> Result<DescriptorSet> {
        let counts = if let Some(count) = descriptor_count {
            vec![count]
        } else {
            vec![]
        };

        let layouts = [layout.raw];
        let info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(self.raw)
            .set_layouts(&layouts);

        let mut variable_info = vk::DescriptorSetVariableDescriptorCountAllocateInfoBuilder::new()
            .descriptor_counts(&counts);

        let info = if descriptor_count.is_some() {
            info.extend_from(&mut variable_info)
        } else {
            info
        };

        let sets =
            unsafe { self.context.device.allocate_descriptor_sets(&info) }.map_err(|err| {
                #[cfg(feature = "tracing")]
                error!("Unable to allocate a descriptor set: {}", err);
                AscheError::VkResult(err)
            })?;

        debug_assert_eq!(sets.len(), 1);
        let set = sets[0];

        self.context
            .set_object_name(name, vk::ObjectType::DESCRIPTOR_SET, set.0)?;

        Ok(DescriptorSet {
            context: self.context.clone(),
            pool: self.raw,
            raw: set,
        })
    }

    /// Frees all descriptor sets allocated from the pool. Invalidates all descriptor sets from the pool.
    pub fn free_sets(&self) -> Result<()> {
        unsafe { self.context.device.reset_descriptor_pool(self.raw, None) }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable reset a descriptor pool: {}", err);
            AscheError::VkResult(err)
        })?;
        Ok(())
    }
}

/// Wraps a descriptor set layout.
#[derive(Debug)]
pub struct DescriptorSetLayout {
    /// The raw Vulkan descriptor set layout.
    pub raw: vk::DescriptorSetLayout,
    context: Arc<Context>,
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_descriptor_set_layout(Some(self.raw), None);
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
#[derive(Debug)]
pub struct DescriptorSet {
    /// The raw Vulkan descriptor pool.
    pub raw: vk::DescriptorSet,
    pool: vk::DescriptorPool,
    context: Arc<Context>,
}

impl DescriptorSet {
    /// Frees the descriptor set. Needs to be created from a pool with the
    /// `vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET` flag set.
    pub fn free(&mut self) -> Result<()> {
        let sets = [self.raw];
        unsafe { self.context.device.free_descriptor_sets(self.pool, &sets) }.map_err(|err| {
            #[cfg(feature = "tracing")]
            error!("Unable to free a descriptor set: {}", err);
            AscheError::VkResult(err)
        })?;

        Ok(())
    }
}
