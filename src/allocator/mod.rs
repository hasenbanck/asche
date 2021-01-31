//! Custom memory allocator.
//!
//! Originally based on the [gpu-allocator](https://github.com/Traverse-Research/gpu-allocator) project.
use std::ffi::c_void;
use std::num::NonZeroU64;

use ash::version::{DeviceV1_0, InstanceV1_0, InstanceV1_1};
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, warn};

use best_fit_allocator::BestFitAllocator;
use dedicated_allocator::DedicatedAllocator;
pub use error::AllocatorError;

type Result<T> = std::result::Result<T, AllocatorError>;

mod best_fit_allocator;
mod dedicated_allocator;
mod error;

/// The configuration descriptor for an allocation.
#[derive(Debug, Clone)]
pub struct AllocationDescriptor {
    /// Name of the allocation, for tracking and debugging purposes.
    pub name: &'static str,
    /// Vulkan memory requirements for an allocation.
    pub requirements: vk::MemoryRequirements,
    /// Location where the memory allocation should be stored.
    pub location: MemoryLocation,
}

/// The location of the memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Store the allocation in GPU only accessible memory. Typically this is the faster GPU resource and this should be
    /// where most of the allocations live.
    GpuOnly,
    /// Memory useful for uploading data to the GPU and potentially for constant buffers.
    CpuToGpu,
    /// Memory useful for reading back data from GPU to CPU.
    GpuToCpu,
}

/// Defines type of the allocation.
#[derive(Debug, Copy, Clone, PartialEq)]
enum AllocationType {
    /// Block is dedicated to a single resource.
    Dedicated,
    /// Block is dedicated for linear resource (buffers and linear images).
    Linear,
    /// Block is dedicated for an optimal textures and images.
    Optimal,
}

/// The memory allocator.
pub struct Allocator {
    memory_types: Vec<MemoryType>,
    device: ash::Device,
    physical_memory_properties: vk::PhysicalDeviceMemoryProperties2,
    buffer_image_granularity: u64,
}

impl Allocator {
    /// Creates a new `Allocator`.
    pub fn new(
        instance: ash::Instance,
        logical_device: ash::Device,
        physical_device: ash::vk::PhysicalDevice,
    ) -> Self {
        let mut properties = vk::PhysicalDeviceMemoryProperties2 {
            ..Default::default()
        };

        unsafe {
            instance.get_physical_device_memory_properties2(physical_device, &mut properties)
        };

        let mem_properties = properties.memory_properties;
        let memory_types = &mem_properties.memory_types[..mem_properties.memory_type_count as _];

        #[cfg(feature = "tracing")]
        {
            debug!("Memory type count: {}", mem_properties.memory_type_count);
            debug!("Memory heap count: {}", mem_properties.memory_heap_count);

            for (i, mem_type) in memory_types.iter().enumerate() {
                let flags = mem_type.property_flags;
                debug!(
                    "Memory type[{}]: prop flags: 0x{:x}, heap[{}]",
                    i,
                    flags.as_raw(),
                    mem_type.heap_index,
                );
            }
            for i in 0..mem_properties.memory_heap_count {
                debug!(
                    "Heap[{}] flags: 0x{:x}, size: {} MiB",
                    i,
                    mem_properties.memory_heaps[i as usize].flags.as_raw(),
                    mem_properties.memory_heaps[i as usize].size / (1024 * 1024)
                );
            }

            let host_visible_not_coherent = memory_types.iter().any(|t| {
                let flags = t.property_flags;
                flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && !flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            });
            if host_visible_not_coherent {
                warn!("There is a memory type that is host visible, but not host coherent.");
            }
        }

        let memory_types: Vec<MemoryType> = memory_types
            .iter()
            .enumerate()
            .map(|(i, mem_type)| MemoryType {
                memory_blocks: Vec::default(),
                memory_properties: mem_type.property_flags,
                memory_type_index: i,
                heap_index: mem_type.heap_index as usize,
                mappable: mem_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                active_general_blocks: 0,
            })
            .collect();

        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let granularity = physical_device_properties.limits.buffer_image_granularity;

        Self {
            memory_types,
            device: logical_device,
            physical_memory_properties: properties,
            buffer_image_granularity: granularity,
        }
    }

    /// Allocates memory for a buffer.
    // TODO https://www.asawicki.info/articles/VK_KHR_dedicated_allocation.php5
    pub fn allocate_memory_for_buffer(
        &mut self,
        desc: &AllocationDescriptor,
    ) -> Result<Allocation> {
        // vkGetBufferMemoryRequirements2()
        // VK_KHR_get_memory_requirements2
        self.allocate_memory(desc, AllocationType::Linear)
    }

    /// Allocates memory for an image.
    // TODO https://www.asawicki.info/articles/VK_KHR_dedicated_allocation.php5
    pub fn allocate_memory_for_image(&mut self, desc: &AllocationDescriptor) -> Result<Allocation> {
        // vkGetImageMemoryRequirements2()
        // VK_KHR_get_memory_requirements2
        // TODO how to decide if an image is linear?
        self.allocate_memory(desc, AllocationType::Optimal)
    }

    fn allocate_memory(
        &mut self,
        desc: &AllocationDescriptor,
        allocation_type: AllocationType,
    ) -> Result<Allocation> {
        let size = desc.requirements.size;
        let alignment = desc.requirements.alignment;

        #[cfg(feature = "tracing")]
        debug!(
            "Allocating \"{}\" of {} bytes with an alignment of {}.",
            desc.name, size, alignment
        );

        if size == 0 || !alignment.is_power_of_two() {
            return Err(AllocatorError::InvalidAllocationDescriptor);
        }

        let mem_loc_preferred_bits = match desc.location {
            MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::CpuToGpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL
            }
            MemoryLocation::GpuToCpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED
            }
        };
        let mut memory_type_index_opt = find_memory_type_index(
            &desc.requirements,
            &self.physical_memory_properties.memory_properties,
            mem_loc_preferred_bits,
        );

        if memory_type_index_opt.is_none() {
            let mem_loc_required_bits = match desc.location {
                MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
                MemoryLocation::CpuToGpu => {
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                }
                MemoryLocation::GpuToCpu => {
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                }
            };

            memory_type_index_opt = find_memory_type_index(
                &desc.requirements,
                &self.physical_memory_properties.memory_properties,
                mem_loc_required_bits,
            );
        }

        let memory_type_index = match memory_type_index_opt {
            Some(x) => x as usize,
            None => return Err(AllocatorError::NoCompatibleMemoryTypeFound),
        };

        let sub_allocation =
            self.memory_types[memory_type_index].allocate(&self.device, desc, allocation_type);

        if desc.location == MemoryLocation::CpuToGpu {
            if sub_allocation.is_err() {
                let mem_loc_preferred_bits =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

                let memory_type_index_opt = find_memory_type_index(
                    &desc.requirements,
                    &self.physical_memory_properties.memory_properties,
                    mem_loc_preferred_bits,
                );

                let memory_type_index = match memory_type_index_opt {
                    Some(x) => x as usize,
                    None => return Err(AllocatorError::NoCompatibleMemoryTypeFound),
                };

                self.memory_types[memory_type_index].allocate(&self.device, desc, allocation_type)
            } else {
                sub_allocation
            }
        } else {
            sub_allocation
        }
    }

    /// Free memory of an allocation.
    pub fn free(&mut self, allocation: Allocation) -> Result<()> {
        #[cfg(feature = "tracing")]
        debug!(
            "Freeing \"{}\".",
            allocation.name.as_deref().unwrap_or("<null>")
        );

        self.memory_types[allocation.memory_type_index].free(allocation, &self.device)?;

        Ok(())
    }

    /// Logs memory leaks as warnings.
    #[cfg(feature = "tracing")]
    fn log_memory_leaks(&self) {
        for (mem_type_i, mem_type) in self.memory_types.iter().enumerate() {
            for (block_i, mem_block) in mem_type.memory_blocks.iter().enumerate() {
                match &mem_block.allocator {
                    SubAllocator::Dedicated(alloc) => alloc.log_memory_leaks(mem_type_i, block_i),
                    SubAllocator::BestFit(alloc) => alloc.log_memory_leaks(mem_type_i, block_i),
                }
            }
        }
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        #[cfg(feature = "tracing")]
        self.log_memory_leaks();

        // Free all remaining memory blocks.
        for mem_type in self.memory_types.iter_mut() {
            for mem_block in mem_type.memory_blocks.iter_mut() {
                mem_block.destroy(&self.device);
            }
        }
    }
}

/// An allocation. Either a dedicated memory block or a sub allocation inside a memory block.
#[derive(Clone, Debug)]
pub struct Allocation {
    chunk_id: NonZeroU64,
    memory_block_index: usize,
    memory_type_index: usize,
    device_memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    mapped_ptr: Option<std::ptr::NonNull<c_void>>,
    name: Option<&'static str>,
}

unsafe impl Send for Allocation {}

impl Allocation {
    /// Returns the `vk::DeviceMemory` object that is backing this allocation.
    /// This memory object can be shared with multiple other allocations and shouldn't be freed (or allocated from)
    /// without this library, because that will lead to undefined behavior.
    ///
    /// # Safety
    /// The result of this function can safely be used to pass into `bind_buffer_memory` (`vkBindBufferMemory`),
    /// `bind_texture_memory` (`vkBindTextureMemory`) etc. It's exposed for this reason. Keep in mind to also
    /// pass `Self::offset()` along to those.
    pub unsafe fn memory(&self) -> vk::DeviceMemory {
        self.device_memory
    }

    /// Returns the offset of the allocation on the vk::DeviceMemory.
    /// When binding the memory to a buffer or image, this offset needs to be supplied as well.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the size of the allocation.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns a valid mapped pointer if the memory is host visible, otherwise it will return None.
    /// The pointer already points to the exact memory region of the sub allocation, so no offset needs to be applied.
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
        self.mapped_ptr
    }

    /// Returns a valid mapped slice if the memory is host visible, otherwise it will return None.
    /// The slice already references the exact memory region of the sub allocation, so no offset needs to be applied.
    pub fn mapped_slice(&self) -> Option<&[u8]> {
        if let Some(ptr) = self.mapped_ptr() {
            unsafe {
                Some(std::slice::from_raw_parts(
                    ptr.as_ptr() as *const _,
                    self.size as usize,
                ))
            }
        } else {
            None
        }
    }

    /// Returns a valid mapped mutable slice if the memory is host visible, otherwise it will return None.
    /// The slice already references the exact memory region of the sub allocation, so no offset needs to be applied.
    pub fn mapped_slice_mut(&mut self) -> Option<&mut [u8]> {
        if let Some(ptr) = self.mapped_ptr() {
            unsafe {
                Some(std::slice::from_raw_parts_mut(
                    ptr.as_ptr() as *mut _,
                    self.size as usize,
                ))
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
enum SubAllocator {
    Dedicated(DedicatedAllocator),
    BestFit(BestFitAllocator),
    //Linear(LinearAllocator),
}

#[derive(Debug)]
struct MemoryBlock {
    device_memory: vk::DeviceMemory,
    size: u64,
    mapped_ptr: *mut c_void,
    allocation_type: AllocationType,
    allocator: SubAllocator,
}

impl MemoryBlock {
    fn new(
        device: &ash::Device,
        size: u64,
        mem_type_index: usize,
        mapped: bool,
        allocation_type: AllocationType,
    ) -> Result<Self> {
        let device_memory = {
            let allocation_flags = vk::MemoryAllocateFlags::DEVICE_ADDRESS;
            let mut flags_info = vk::MemoryAllocateFlagsInfo::builder().flags(allocation_flags);

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(size)
                .memory_type_index(mem_type_index as u32)
                .push_next(&mut flags_info);

            unsafe { device.allocate_memory(&alloc_info, None) }
                .map_err(|_| AllocatorError::OutOfMemory)?
        };

        let mapped_ptr = if mapped {
            unsafe {
                device.map_memory(
                    device_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|_| {
                unsafe { device.free_memory(device_memory, None) };
                AllocatorError::FailedToMap
            })?
        } else {
            std::ptr::null_mut()
        };

        let allocator: SubAllocator = match allocation_type {
            AllocationType::Dedicated => SubAllocator::Dedicated(DedicatedAllocator::new(size)),
            AllocationType::Linear => SubAllocator::BestFit(BestFitAllocator::new(size)),
            AllocationType::Optimal => SubAllocator::BestFit(BestFitAllocator::new(size)),
        };

        Ok(Self {
            device_memory,
            size,
            mapped_ptr,
            allocation_type,
            allocator,
        })
    }

    fn destroy(&mut self, device: &ash::Device) {
        if !self.mapped_ptr.is_null() {
            unsafe { device.unmap_memory(self.device_memory) };
        }

        unsafe { device.free_memory(self.device_memory, None) };
    }
}

#[derive(Debug)]
struct MemoryType {
    memory_blocks: Vec<MemoryBlock>,
    memory_properties: vk::MemoryPropertyFlags,
    memory_type_index: usize,
    heap_index: usize,
    mappable: bool,
    active_general_blocks: usize,
}

impl MemoryType {
    fn allocate(
        &mut self,
        device: &ash::Device,
        desc: &AllocationDescriptor,
        allocation_type: AllocationType,
    ) -> Result<Allocation> {
        let memory_block_size = 64 * 1024 * 1024; // 64 MiB

        let size = desc.requirements.size;
        let alignment = desc.requirements.alignment;

        // Create a dedicated block for large memory allocations
        if size > memory_block_size || allocation_type == AllocationType::Dedicated {
            let mem_block = MemoryBlock::new(
                device,
                size,
                self.memory_type_index,
                self.mappable,
                AllocationType::Dedicated,
            )?;

            self.memory_blocks.push(mem_block);
            let block_index = self.memory_blocks.len() - 1;
            let mem_block = &mut self.memory_blocks[block_index];

            let (offset, chunk_id) = match mem_block.allocator {
                SubAllocator::Dedicated(ref mut alloc) => alloc.allocate(size, desc.name)?,
                _ => return Err(AllocatorError::Internal("unexpected allocator")),
            };

            return Ok(Allocation {
                chunk_id,
                memory_block_index: block_index,
                memory_type_index: self.memory_type_index as usize,
                device_memory: mem_block.device_memory,
                offset,
                size,
                mapped_ptr: std::ptr::NonNull::new(mem_block.mapped_ptr),
                name: Some(desc.name),
            });
        }

        // Try to fo the allocation in an existing memory block.
        for (mem_block_i, mem_block) in self
            .memory_blocks
            .iter_mut()
            .filter(|block| block.allocation_type == allocation_type)
            .enumerate()
        {
            let allocation = match mem_block.allocator {
                SubAllocator::BestFit(ref mut alloc) => alloc.allocate(size, alignment, desc.name),
                _ => return Err(AllocatorError::Internal("unexpected allocator")),
            };

            match allocation {
                Ok((offset, chunk_id)) => {
                    let mapped_ptr = get_mapped_ptr(mem_block, offset);
                    return Ok(Allocation {
                        chunk_id,
                        memory_block_index: mem_block_i,
                        memory_type_index: self.memory_type_index as usize,
                        device_memory: mem_block.device_memory,
                        offset,
                        size,
                        mapped_ptr,
                        name: Some(desc.name),
                    });
                }
                Err(err) => match err {
                    AllocatorError::OutOfMemory => { /* Block is full, continue search */ }
                    _ => return Err(err),
                },
            }
        }

        // We found no suitable allocation, so we need to create a new block.
        let new_memory_block = MemoryBlock::new(
            device,
            memory_block_size,
            self.memory_type_index,
            self.mappable,
            allocation_type,
        )?;

        self.memory_blocks.push(new_memory_block);
        let new_block_index = self.memory_blocks.len() - 1;

        self.active_general_blocks += 1;

        let mem_block = &mut self.memory_blocks[new_block_index];
        let allocation = match mem_block.allocator {
            SubAllocator::BestFit(ref mut alloc) => alloc.allocate(size, alignment, desc.name),
            _ => return Err(AllocatorError::Internal("unexpected allocator")),
        };
        let (offset, chunk_id) = match allocation {
            Ok(value) => value,
            Err(err) => {
                return match err {
                    AllocatorError::OutOfMemory => Err(AllocatorError::Internal(
                        "allocation that must succeed failed",
                    )),
                    _ => Err(err),
                };
            }
        };

        let mapped_ptr = get_mapped_ptr(mem_block, offset);

        Ok(Allocation {
            chunk_id,
            memory_block_index: new_block_index,
            memory_type_index: self.memory_type_index as usize,
            device_memory: mem_block.device_memory,
            offset,
            size,
            mapped_ptr,
            name: Some(desc.name),
        })
    }

    fn free(&mut self, allocation: Allocation, device: &ash::Device) -> Result<()> {
        let block_idx = allocation.memory_block_index;

        let mem_block = &mut self.memory_blocks[block_idx];
        let chunk_id = allocation.chunk_id;

        match &mut mem_block.allocator {
            SubAllocator::Dedicated(_) => {
                let mut block = self.memory_blocks.remove(block_idx);
                block.destroy(device);
            }
            SubAllocator::BestFit(ref mut alloc) => {
                alloc.free(chunk_id)?;
                if alloc.allocated() == 0 {
                    let mut block = self.memory_blocks.remove(block_idx);
                    block.destroy(device);
                }
            }
        }

        Ok(())
    }
}

fn get_mapped_ptr(mem_block: &mut MemoryBlock, offset: u64) -> Option<std::ptr::NonNull<c_void>> {
    if !mem_block.mapped_ptr.is_null() {
        let offset_ptr = unsafe { mem_block.mapped_ptr.add(offset as usize) };
        std::ptr::NonNull::new(offset_ptr)
    } else {
        None
    }
}

fn find_memory_type_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags.contains(flags)
        })
        .map(|(index, _memory_type)| index as _)
}
