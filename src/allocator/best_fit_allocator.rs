use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

#[cfg(feature = "tracing")]
use tracing::warn;

use super::{AllocatorError, Result};

fn align_down(val: u64, alignment: u64) -> u64 {
    val & !(alignment - 1u64)
}

fn align_up(val: u64, alignment: u64) -> u64 {
    align_down(val + alignment - 1u64, alignment)
}

#[derive(Debug)]
struct MemoryChunk {
    chunk_id: NonZeroU64,
    size: u64,
    offset: u64,
    is_free: bool,
    name: Option<&'static str>,
    next: Option<NonZeroU64>,
    prev: Option<NonZeroU64>,
}

/// A simple free list allocator.
#[derive(Debug)]
pub(crate) struct BestFitAllocator {
    size: u64,
    allocated: u64,
    chunk_id_counter: u64,
    chunks: HashMap<NonZeroU64, MemoryChunk>,
    free_chunks: HashSet<NonZeroU64>,
}

impl BestFitAllocator {
    pub(crate) fn new(size: u64) -> BestFitAllocator {
        let initial_chunk_id = NonZeroU64::new(1).unwrap();

        let mut chunks = HashMap::default();
        chunks.insert(
            initial_chunk_id,
            MemoryChunk {
                chunk_id: initial_chunk_id,
                size,
                offset: 0,
                is_free: true,
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

    /// Generates a new unique chunk ID.
    fn get_new_chunk_id(&mut self) -> Result<NonZeroU64> {
        if self.chunk_id_counter == std::u64::MAX {
            // End of chunk id counter reached, no more allocations are possible.
            return Err(AllocatorError::OutOfMemory);
        }

        let id = self.chunk_id_counter;
        self.chunk_id_counter += 1;
        NonZeroU64::new(id).ok_or(AllocatorError::Internal("new chunk id was 0"))
    }
    /// Finds the specified chunk_id in the list of free chunks and removes if from the list.
    fn remove_id_from_free_list(&mut self, chunk_id: NonZeroU64) {
        self.free_chunks.remove(&chunk_id);
    }
    /// Merges two adjacent chunks. Right chunk will be merged into the left chunk.
    fn merge_free_chunks(&mut self, chunk_left: NonZeroU64, chunk_right: NonZeroU64) -> Result<()> {
        // Gather data from right chunk and remove it.
        let (right_size, right_next) = {
            let chunk = self
                .chunks
                .remove(&chunk_right)
                .ok_or(AllocatorError::Internal(
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
                .ok_or(AllocatorError::Internal(
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
                .ok_or(AllocatorError::Internal(
                    "chunk ID not present in chunk list",
                ))?;
            chunk.prev = Some(chunk_left);
        }

        Ok(())
    }
}

impl BestFitAllocator {
    pub(crate) fn allocate(
        &mut self,
        size: u64,
        alignment: u64,
        name: &'static str,
    ) -> Result<(u64, NonZeroU64)> {
        let free_size = self.size - self.allocated;
        if size > free_size {
            return Err(AllocatorError::OutOfMemory);
        }

        let mut best_fit_id: Option<NonZeroU64> = None;
        let mut best_offset = 0u64;
        let mut best_aligned_size = 0u64;
        let mut best_chunk_size = 0u64;

        for current_chunk_id in self.free_chunks.iter() {
            let current_chunk =
                self.chunks
                    .get(&current_chunk_id)
                    .ok_or(AllocatorError::Internal(
                        "chunk ID in free list is not present in chunk list",
                    ))?;

            if current_chunk.size < size {
                continue;
            }

            let offset = align_up(current_chunk.offset, alignment);
            let padding = offset - current_chunk.offset;
            let aligned_size = padding + size;

            if aligned_size > current_chunk.size {
                continue;
            }

            if best_fit_id.is_none() || current_chunk.size < best_chunk_size {
                best_fit_id = Some(*current_chunk_id);
                best_aligned_size = aligned_size;
                best_offset = offset;

                best_chunk_size = current_chunk.size;
            };
        }

        let first_fit_id = best_fit_id.ok_or(AllocatorError::OutOfMemory)?;

        let chunk_id = if best_chunk_size > best_aligned_size {
            let new_chunk_id = self.get_new_chunk_id()?;

            let new_chunk = {
                let free_chunk = self
                    .chunks
                    .get_mut(&first_fit_id)
                    .ok_or(AllocatorError::Internal("chunk ID must be in chunk list"))?;
                let new_chunk = MemoryChunk {
                    chunk_id: new_chunk_id,
                    size: best_aligned_size,
                    offset: free_chunk.offset,
                    is_free: true,
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
                    .ok_or(AllocatorError::Internal("invalid previous chunk reference"))?;
                prev_chunk.next = Some(new_chunk.chunk_id);
            }

            self.chunks.insert(new_chunk_id, new_chunk);

            new_chunk_id
        } else {
            let chunk = self
                .chunks
                .get_mut(&first_fit_id)
                .ok_or(AllocatorError::Internal("invalid chunk reference"))?;

            chunk.name = Some(name);

            self.remove_id_from_free_list(first_fit_id);

            first_fit_id
        };

        self.allocated += best_aligned_size;

        Ok((best_offset, chunk_id))
    }

    pub(crate) fn free(&mut self, chunk_id: NonZeroU64) -> Result<()> {
        let (next_id, prev_id) = {
            let chunk = self
                .chunks
                .get_mut(&chunk_id)
                .ok_or(AllocatorError::Internal(
                    "attempting to free chunk that is not in chunk list",
                ))?;
            chunk.is_free = true;
            chunk.name = None;

            self.allocated -= chunk.size;

            self.free_chunks.insert(chunk.chunk_id);

            (chunk.next, chunk.prev)
        };

        if let Some(next_id) = next_id {
            if self.chunks[&next_id].is_free {
                self.merge_free_chunks(chunk_id, next_id)?;
            }
        }

        if let Some(prev_id) = prev_id {
            if self.chunks[&prev_id].is_free {
                self.merge_free_chunks(prev_id, chunk_id)?;
            }
        }
        Ok(())
    }

    pub(crate) fn log_memory_leaks(&self, memory_type_index: usize, memory_block_index: usize) {
        for (chunk_id, chunk) in self.chunks.iter() {
            if chunk.is_free {
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
        name: {},
    }}
}}"#,
                memory_type_index, memory_block_index, chunk_id, chunk.size, chunk.offset, name,
            );
        }
    }

    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    pub(crate) fn allocated(&self) -> u64 {
        self.allocated
    }
}
