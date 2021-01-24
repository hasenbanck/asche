use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

#[cfg(feature = "tracing")]
use tracing::warn;

use super::{AllocationError, AllocationType, Result, SubAllocation, SubAllocator};

fn align_down(val: u64, alignment: u64) -> u64 {
    val & !(alignment - 1u64)
}

fn align_up(val: u64, alignment: u64) -> u64 {
    align_down(val + alignment - 1u64, alignment)
}

#[derive(Debug)]
pub(crate) struct MemoryChunk {
    pub(crate) chunk_id: NonZeroU64,
    pub(crate) size: u64,
    pub(crate) offset: u64,
    pub(crate) allocation_type: AllocationType,
    pub(crate) name: Option<&'static str>,
    next: Option<NonZeroU64>,
    prev: Option<NonZeroU64>,
}

/// A simple free list allocator.
#[derive(Debug)]
pub(crate) struct FreeListAllocator {
    size: u64,
    allocated: u64,
    pub(crate) chunk_id_counter: u64,
    pub(crate) chunks: HashMap<NonZeroU64, MemoryChunk>,
    free_chunks: HashSet<NonZeroU64>,
}

/// Test if two sub allocations will overlap the same page.
fn is_on_same_page(offset_a: u64, size_a: u64, offset_b: u64, page_size: u64) -> bool {
    let end_a = offset_a + size_a - 1;
    let end_page_a = align_down(end_a, page_size);
    let start_b = offset_b;
    let start_page_b = align_down(start_b, page_size);

    end_page_a == start_page_b
}

/// Test if two allocation types will be conflicting or not.
fn has_granularity_conflict(lhs: AllocationType, rhs: AllocationType) -> bool {
    if lhs == AllocationType::Free || rhs == AllocationType::Free {
        return false;
    }

    lhs != rhs
}

impl FreeListAllocator {
    pub(crate) fn new(size: u64) -> FreeListAllocator {
        let initial_chunk_id = NonZeroU64::new(1).unwrap();

        let mut chunks = HashMap::default();
        chunks.insert(
            initial_chunk_id,
            MemoryChunk {
                chunk_id: initial_chunk_id,
                size,
                offset: 0,
                allocation_type: AllocationType::Free,
                name: None,
                prev: None,
                next: None,
            },
        );

        let mut free_chunks = HashSet::default();
        free_chunks.insert(initial_chunk_id);

        Self {
            size,
            allocated: 0,
            // 0 is not allowed as a chunk ID, 1 is used by the initial chunk, next chunk is going to be 2.
            // The system well take the counter as the ID, and the increment the counter.
            chunk_id_counter: 2,
            chunks,
            free_chunks,
        }
    }

    /// Generates a new unique chunk ID
    fn get_new_chunk_id(&mut self) -> Result<NonZeroU64> {
        if self.chunk_id_counter == std::u64::MAX {
            // End of chunk id counter reached, no more allocations are possible.
            return Err(AllocationError::OutOfMemory);
        }

        let id = self.chunk_id_counter;
        self.chunk_id_counter += 1;
        NonZeroU64::new(id).ok_or(AllocationError::Internal("new chunk id was 0"))
    }
    /// Finds the specified chunk_id in the list of free chunks and removes if from the list
    fn remove_id_from_free_list(&mut self, chunk_id: NonZeroU64) {
        self.free_chunks.remove(&chunk_id);
    }
    /// Merges two adjacent chunks. Right chunk will be merged into the left chunk
    fn merge_free_chunks(&mut self, chunk_left: NonZeroU64, chunk_right: NonZeroU64) -> Result<()> {
        // Gather data from right chunk and remove it.
        let (right_size, right_next) = {
            let chunk = self
                .chunks
                .remove(&chunk_right)
                .ok_or(AllocationError::Internal(
                    "chunk ID not present in chunk list",
                ))?;
            self.remove_id_from_free_list(chunk.chunk_id);

            (chunk.size, chunk.next)
        };

        // Merge into left chunk.
        {
            let chunk = self
                .chunks
                .get_mut(&chunk_left)
                .ok_or(AllocationError::Internal(
                    "chunk ID not present in chunk list",
                ))?;
            chunk.next = right_next;
            chunk.size += right_size;
        }

        // Patch pointers.
        if let Some(right_next) = right_next {
            let chunk = self
                .chunks
                .get_mut(&right_next)
                .ok_or(AllocationError::Internal(
                    "chunk ID not present in chunk list",
                ))?;
            chunk.prev = Some(chunk_left);
        }

        Ok(())
    }
}

impl SubAllocator for FreeListAllocator {
    fn allocate(
        &mut self,
        size: u64,
        alignment: u64,
        allocation_type: AllocationType,
        granularity: u64,
        name: &'static str,
    ) -> Result<(u64, NonZeroU64)> {
        let free_size = self.size - self.allocated;
        if size > free_size {
            return Err(AllocationError::OutOfMemory);
        }

        let mut best_fit_id: Option<NonZeroU64> = None;
        let mut best_offset = 0u64;
        let mut best_aligned_size = 0u64;
        let mut best_chunk_size = 0u64;

        for current_chunk_id in self.free_chunks.iter() {
            let current_chunk =
                self.chunks
                    .get(&current_chunk_id)
                    .ok_or(AllocationError::Internal(
                        "chunk ID in free list is not present in chunk list",
                    ))?;

            if current_chunk.size < size {
                continue;
            }

            let mut offset = align_up(current_chunk.offset, alignment);

            if let Some(prev_idx) = current_chunk.prev {
                let previous = self.chunks.get(&prev_idx).ok_or(AllocationError::Internal(
                    "invalid previous chunk reference",
                ))?;
                if is_on_same_page(previous.offset, previous.size, offset, granularity)
                    && has_granularity_conflict(previous.allocation_type, allocation_type)
                {
                    offset = align_up(offset, granularity);
                }
            }

            let padding = offset - current_chunk.offset;
            let aligned_size = padding + size;

            if aligned_size > current_chunk.size {
                continue;
            }

            if let Some(next_idx) = current_chunk.next {
                let next = self
                    .chunks
                    .get(&next_idx)
                    .ok_or(AllocationError::Internal("invalid next chunk reference"))?;
                if is_on_same_page(offset, size, next.offset, granularity)
                    && has_granularity_conflict(allocation_type, next.allocation_type)
                {
                    continue;
                }
            }

            if best_fit_id.is_none() || current_chunk.size < best_chunk_size {
                best_fit_id = Some(*current_chunk_id);
                best_aligned_size = aligned_size;
                best_offset = offset;

                best_chunk_size = current_chunk.size;
            };
        }

        let first_fit_id = best_fit_id.ok_or(AllocationError::OutOfMemory)?;

        let chunk_id = if best_chunk_size > best_aligned_size {
            let new_chunk_id = self.get_new_chunk_id()?;

            let new_chunk = {
                let free_chunk = self
                    .chunks
                    .get_mut(&first_fit_id)
                    .ok_or(AllocationError::Internal("chunk ID must be in chunk list"))?;
                let new_chunk = MemoryChunk {
                    chunk_id: new_chunk_id,
                    size: best_aligned_size,
                    offset: free_chunk.offset,
                    allocation_type,
                    name: Some(name),
                    prev: free_chunk.prev,
                    next: Some(first_fit_id),
                };

                free_chunk.prev = Some(new_chunk.chunk_id);
                free_chunk.offset += best_aligned_size;
                free_chunk.size -= best_aligned_size;
                new_chunk
            };

            if let Some(prev_id) = new_chunk.prev {
                let prev_chunk = self
                    .chunks
                    .get_mut(&prev_id)
                    .ok_or(AllocationError::Internal(
                        "invalid previous chunk reference",
                    ))?;
                prev_chunk.next = Some(new_chunk.chunk_id);
            }

            self.chunks.insert(new_chunk_id, new_chunk);

            new_chunk_id
        } else {
            let chunk = self
                .chunks
                .get_mut(&first_fit_id)
                .ok_or(AllocationError::Internal("invalid chunk reference"))?;

            chunk.allocation_type = allocation_type;
            chunk.name = Some(name);

            self.remove_id_from_free_list(first_fit_id);

            first_fit_id
        };

        self.allocated += best_aligned_size;

        Ok((best_offset, chunk_id))
    }

    fn free(&mut self, sub_allocation: SubAllocation) -> Result<()> {
        let chunk_id = sub_allocation
            .chunk_id
            .ok_or(AllocationError::Internal("chunk ID must be a valid value"))?;

        let (next_id, prev_id) = {
            let chunk = self
                .chunks
                .get_mut(&chunk_id)
                .ok_or(AllocationError::Internal(
                    "attempting to free chunk that is not in chunk list",
                ))?;
            chunk.allocation_type = AllocationType::Free;
            chunk.name = None;

            self.allocated -= chunk.size;

            self.free_chunks.insert(chunk.chunk_id);

            (chunk.next, chunk.prev)
        };

        if let Some(next_id) = next_id {
            if self.chunks[&next_id].allocation_type == AllocationType::Free {
                self.merge_free_chunks(chunk_id, next_id)?;
            }
        }

        if let Some(prev_id) = prev_id {
            if self.chunks[&prev_id].allocation_type == AllocationType::Free {
                self.merge_free_chunks(prev_id, chunk_id)?;
            }
        }
        Ok(())
    }

    fn log_memory_leaks(&self, memory_type_index: usize, memory_block_index: usize) {
        for (chunk_id, chunk) in self.chunks.iter() {
            if chunk.allocation_type == AllocationType::Free {
                continue;
            }
            let name = chunk.name.as_ref().unwrap_or(&"");

            warn!(
                r#"leak detected: {{
    memory type: {}
    memory block: {}
    chunk: {{
        chunk_id: {},
        size: 0x{:x},
        offset: 0x{:x},
        allocation_type: {:?},
        name: {},
    }}
}}"#,
                memory_type_index,
                memory_block_index,
                chunk_id,
                chunk.size,
                chunk.offset,
                chunk.allocation_type,
                name,
            );
        }
    }

    fn supports_general_allocations(&self) -> bool {
        true
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn allocated(&self) -> u64 {
        self.allocated
    }
}
