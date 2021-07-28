use std::sync::Arc;

use crate::{context::Context, Lifetime};

/// The internal memory allocator.
#[derive(Debug)]
pub(crate) struct MemoryAllocator<LT: Lifetime> {
    /// The memory allocator.
    pub(crate) allocator: vk_alloc::Allocator<LT>,
    context: Arc<Context>,
}

impl<LT: Lifetime> MemoryAllocator<LT> {
    /// Creates a new memory allocator.
    pub(crate) fn new(allocator: vk_alloc::Allocator<LT>, context: Arc<Context>) -> Self {
        Self { allocator, context }
    }
}

impl<LT: Lifetime> Drop for MemoryAllocator<LT> {
    fn drop(&mut self) {
        // All images & buffers have an Arc on the memory allocator, so this is safe.
        unsafe {
            self.allocator.cleanup(&self.context.device);
        }
    }
}
