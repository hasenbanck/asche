use std::num::NonZeroU64;

#[cfg(feature = "tracing")]
use tracing::warn;

use super::{AllocatorError, Result};

/// Allocates a dedicated blob of memory for the given resource.
#[derive(Debug)]
pub(crate) struct DedicatedAllocator {
    size: u64,
    name: Option<String>,
}

impl DedicatedAllocator {
    pub(crate) fn new(size: u64) -> Self {
        Self { size, name: None }
    }
}

impl DedicatedAllocator {
    pub(crate) fn allocate(&mut self, size: u64, name: &'static str) -> Result<(u64, NonZeroU64)> {
        if self.size != size {
            return Err(AllocatorError::Internal(
                "DedicatedBlockAllocator size must match allocation size.",
            ));
        }
        self.name = Some(name.to_string());

        let dummy_id = NonZeroU64::new(1).unwrap();
        Ok((0, dummy_id))
    }

    pub(crate) fn log_memory_leaks(&self, memory_type_index: usize, memory_block_index: usize) {
        let empty = "".to_string();
        let name = self.name.as_ref().unwrap_or(&empty);

        warn!(
            r#"leak detected: {{
    memory type: {}
    memory block: {}
    dedicated allocation: {{
        size: 0x{:x},
        name: {},
    }}
}}"#,
            memory_type_index, memory_block_index, self.size, name
        )
    }

    pub(crate) fn size(&self) -> u64 {
        self.size
    }
}
