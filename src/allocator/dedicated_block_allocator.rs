use std::num::NonZeroU64;

#[cfg(feature = "tracing")]
use tracing::warn;

use super::{AllocationError, AllocationType, Result, SubAllocation, SubAllocator};

/// Allocates a dedicated blob of memory for the given resource.
#[derive(Debug)]
pub(crate) struct DedicatedBlockAllocator {
    size: u64,
    allocated: u64,
    name: Option<String>,
    backtrace: Option<String>,
}

impl DedicatedBlockAllocator {
    pub(crate) fn new(size: u64) -> Self {
        Self {
            size,
            allocated: 0,
            name: None,
            backtrace: None,
        }
    }
}

impl SubAllocator for DedicatedBlockAllocator {
    fn allocate(
        &mut self,
        size: u64,
        _alignment: u64,
        _allocation_type: AllocationType,
        _granularity: u64,
        name: &'static str,
    ) -> Result<(u64, NonZeroU64)> {
        if self.allocated != 0 {
            return Err(AllocationError::OutOfMemory);
        }

        if self.size != size {
            return Err(AllocationError::Internal(
                "DedicatedBlockAllocator size must match allocation size.",
            ));
        }

        self.allocated = size;
        self.name = Some(name.to_string());

        let dummy_id = NonZeroU64::new(1).unwrap();
        Ok((0, dummy_id))
    }

    fn free(&mut self, sub_allocation: SubAllocation) -> Result<()> {
        if sub_allocation.chunk_id != NonZeroU64::new(1) {
            Err(AllocationError::Internal("chunk ID must be 1"))
        } else {
            self.allocated = 0;
            Ok(())
        }
    }

    fn log_memory_leaks(&self, memory_type_index: usize, memory_block_index: usize) {
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

    fn supports_general_allocations(&self) -> bool {
        false
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn allocated(&self) -> u64 {
        self.allocated
    }
}
