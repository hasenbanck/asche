use std::ffi::c_void;
use std::sync::Arc;

use erupt::vk;
#[cfg(feature = "tracing")]
use tracing::error;

use crate::{AscheError, Context, Result};

/// Wraps a Query Pool.
pub struct QueryPool {
    /// The raw Vulkan query pool.
    pub raw: vk::QueryPool,
    context: Arc<Context>,
}

impl QueryPool {
    pub(crate) fn new(raw: vk::QueryPool, context: Arc<Context>) -> Self {
        Self { raw, context }
    }

    /// Copy results of queries in a query pool to a host memory region.
    ///
    /// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetQueryPoolResults.html
    pub fn results(
        &self,
        first_query: u32,
        query_count: u32,
        data: &mut [u8],
        stride: u64,
        flags: Option<vk::QueryResultFlags>,
    ) -> Result<()> {
        unsafe {
            self.context.device.get_query_pool_results(
                self.raw,
                first_query,
                query_count,
                data.len(),
                data.as_mut_ptr() as *mut c_void,
                stride,
                flags,
            )
        }
        .map_err(|err| {
            #[cfg(feature = "tracing")]
            error!(
                "Unable to copy results of queries in a query pool to a host memory region: {}",
                err
            );
            AscheError::VkResult(err)
        })
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_query_pool(Some(self.raw), None);
        }
    }
}
