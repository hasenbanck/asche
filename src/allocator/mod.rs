//! This module is a hard copy of the [gpu-allocator](https://github.com/Traverse-Research/gpu-allocator) project.
use std::ffi::c_void;
use std::num::NonZeroU64;

use ash::version::{DeviceV1_0, InstanceV1_0, InstanceV1_1};
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, warn};

use dedicated_block_allocator::DedicatedBlockAllocator;
pub use error::AllocationError;
use free_list_allocator::FreeListAllocator;

mod error;

type Result<T> = std::result::Result<T, AllocationError>;

mod dedicated_block_allocator;
mod free_list_allocator;

/// The configuration descriptor for an allocation.
#[derive(Clone, Debug)]
pub struct AllocationDescriptor {
    /// Name of the allocation, for tracking and debugging purposes.
    pub name: &'static str,
    /// Vulkan memory requirements for an allocation.
    pub requirements: vk::MemoryRequirements,
    /// Location where the memory allocation should be stored.
    pub location: MemoryLocation,
}

/// The location of the memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Store the allocation in GPU only accessible memory. Typically this is the faster GPU resource and this should be
    /// where most of the allocations live.
    GpuOnly,
    /// Memory useful for uploading data to the GPU and potentially for constant buffers.
    CpuToGpu,
    /// Memory useful for reading back data from GPU to CPU.
    GpuToCpu,
}

trait SubAllocator: std::fmt::Debug {
    /// Create a new sub allocation. Returns the offset and the chunk ID of the sub allocation.
    fn allocate(
        &mut self,
        size: u64,
        alignment: u64,
        allocation_type: AllocationType,
        granularity: u64,
        name: &'static str,
    ) -> Result<(u64, NonZeroU64)>;

    /// Free a sub allocation.
    fn free(&mut self, chunk_id: NonZeroU64) -> Result<()>;

    /// Logs memory leaks as warnings.
    fn log_memory_leaks(&self, memory_type_index: usize, memory_block_index: usize);

    fn supports_general_allocations(&self) -> bool;

    fn size(&self) -> u64;

    fn allocated(&self) -> u64;

    /// Reports how much memory is available in this sub allocator
    fn available_memory(&self) -> u64 {
        self.size() - self.allocated()
    }

    /// Reports if the sub allocator is empty (having no allocations).
    fn is_empty(&self) -> bool {
        self.allocated() == 0
    }
}

/// A sub allocation.
#[derive(Clone, Debug)]
pub struct SubAllocation {
    chunk_id: Option<NonZeroU64>,
    memory_block_index: usize,
    memory_type_index: usize,
    device_memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    mapped_ptr: Option<std::ptr::NonNull<c_void>>,
    name: Option<&'static str>,
}

unsafe impl Send for SubAllocation {}

impl SubAllocation {
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

    /// Returns the size of the allocation
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

    /// Reports if sub allocation is unallocated.
    pub fn is_null(&self) -> bool {
        self.chunk_id.is_none()
    }
}

impl Default for SubAllocation {
    fn default() -> Self {
        Self {
            chunk_id: None,
            memory_block_index: !0,
            memory_type_index: !0,
            device_memory: vk::DeviceMemory::null(),
            offset: 0,
            size: 0,
            mapped_ptr: None,
            name: None,
        }
    }
}

/// Tracks the usage status of an allocation.
#[derive(PartialEq, Copy, Clone, Debug)]
#[repr(u8)]
pub(crate) enum AllocationType {
    /// Free to use.
    Free,
    /// Linear memory block (buffer / linear texture).
    Linear,
    /// Nonlinear memory block (regular texture).
    NonLinear,
}

#[derive(Debug)]
struct MemoryBlock {
    device_memory: vk::DeviceMemory,
    size: u64,
    mapped_ptr: *mut c_void,
    sub_allocator: Box<dyn SubAllocator>,
}

impl MemoryBlock {
    fn new(
        device: &ash::Device,
        size: u64,
        mem_type_index: usize,
        mapped: bool,
        dedicated: bool,
    ) -> Result<Self> {
        let device_memory = {
            let allocation_flags = vk::MemoryAllocateFlags::DEVICE_ADDRESS;
            let mut flags_info = vk::MemoryAllocateFlagsInfo::builder().flags(allocation_flags);

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(size)
                .memory_type_index(mem_type_index as u32)
                .push_next(&mut flags_info);

            unsafe { device.allocate_memory(&alloc_info, None) }
                .map_err(|_| AllocationError::OutOfMemory)?
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
                AllocationError::FailedToMap
            })?
        } else {
            std::ptr::null_mut()
        };

        let sub_allocator: Box<dyn SubAllocator> = if dedicated {
            Box::new(DedicatedBlockAllocator::new(size))
        } else {
            Box::new(FreeListAllocator::new(size))
        };

        Ok(Self {
            device_memory,
            size,
            mapped_ptr,
            sub_allocator,
        })
    }

    fn destroy(self, device: &ash::Device) {
        if !self.mapped_ptr.is_null() {
            unsafe { device.unmap_memory(self.device_memory) };
        }

        unsafe { device.free_memory(self.device_memory, None) };
    }
}

#[derive(Debug)]
struct MemoryType {
    memory_blocks: Vec<Option<MemoryBlock>>,
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
        linear: bool,
        dedicated: bool,
        granularity: u64,
    ) -> Result<SubAllocation> {
        let allocation_type = if linear {
            AllocationType::Linear
        } else {
            AllocationType::NonLinear
        };

        let memory_block_size = if self
            .memory_properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            // Host memory block size = 64 MiB
            64 * 1024 * 1024
        } else {
            // Device memory block size = 256 MiB
            256 * 1024 * 1024
        };

        let size = desc.requirements.size;
        let alignment = desc.requirements.alignment;

        // Create a dedicated block for large memory allocations
        if size > memory_block_size || dedicated {
            let mem_block =
                MemoryBlock::new(device, size, self.memory_type_index, self.mappable, true)?;

            let mut block_index = None;
            for (i, block) in self.memory_blocks.iter().enumerate() {
                if block.is_none() {
                    block_index = Some(i);
                    break;
                }
            }

            let block_index = match block_index {
                Some(i) => {
                    self.memory_blocks[i].replace(mem_block);
                    i
                }
                None => {
                    self.memory_blocks.push(Some(mem_block));
                    self.memory_blocks.len() - 1
                }
            };

            let mem_block = self.memory_blocks[block_index]
                .as_mut()
                .ok_or(AllocationError::Internal("memory block must be Some"))?;

            let (offset, chunk_id) = mem_block.sub_allocator.allocate(
                size,
                alignment,
                allocation_type,
                granularity,
                desc.name,
            )?;

            return Ok(SubAllocation {
                chunk_id: Some(chunk_id),
                memory_block_index: block_index,
                memory_type_index: self.memory_type_index as usize,
                device_memory: mem_block.device_memory,
                offset,
                size,
                mapped_ptr: std::ptr::NonNull::new(mem_block.mapped_ptr),
                name: Some(desc.name),
            });
        }

        let mut empty_block_index = None;
        for (mem_block_i, mem_block) in self.memory_blocks.iter_mut().enumerate().rev() {
            if let Some(mem_block) = mem_block {
                let allocation = mem_block.sub_allocator.allocate(
                    size,
                    alignment,
                    allocation_type,
                    granularity,
                    desc.name,
                );

                match allocation {
                    Ok((offset, chunk_id)) => {
                        let mapped_ptr = get_mapped_ptr(mem_block, offset);
                        return Ok(SubAllocation {
                            chunk_id: Some(chunk_id),
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
                        AllocationError::OutOfMemory => {} // Block is full, continue search.
                        _ => return Err(err),              // Unhandled error, return.
                    },
                }
            } else if empty_block_index == None {
                empty_block_index = Some(mem_block_i);
            }
        }

        let new_memory_block = MemoryBlock::new(
            device,
            memory_block_size,
            self.memory_type_index,
            self.mappable,
            false,
        )?;

        let new_block_index = if let Some(block_index) = empty_block_index {
            self.memory_blocks[block_index] = Some(new_memory_block);
            block_index
        } else {
            self.memory_blocks.push(Some(new_memory_block));
            self.memory_blocks.len() - 1
        };

        self.active_general_blocks += 1;

        let mem_block = self.memory_blocks[new_block_index]
            .as_mut()
            .ok_or(AllocationError::Internal("memory block must be Some"))?;
        let allocation = mem_block.sub_allocator.allocate(
            size,
            alignment,
            allocation_type,
            granularity,
            desc.name,
        );
        let (offset, chunk_id) = match allocation {
            Ok(value) => value,
            Err(err) => {
                return match err {
                    AllocationError::OutOfMemory => Err(AllocationError::Internal(
                        "allocation that must succeed failed",
                    )),
                    _ => Err(err),
                };
            }
        };

        let mapped_ptr = get_mapped_ptr(mem_block, offset);

        Ok(SubAllocation {
            chunk_id: Some(chunk_id),
            memory_block_index: new_block_index,
            memory_type_index: self.memory_type_index as usize,
            device_memory: mem_block.device_memory,
            offset,
            size,
            mapped_ptr,
            name: Some(desc.name),
        })
    }

    fn free(&mut self, sub_allocation: SubAllocation, device: &ash::Device) -> Result<()> {
        let block_idx = sub_allocation.memory_block_index;

        let mem_block = self.memory_blocks[block_idx]
            .as_mut()
            .ok_or(AllocationError::Internal("memory block must be Some"))?;

        let chunk_id = sub_allocation
            .chunk_id
            .ok_or(AllocationError::Internal("chunk ID must be be Some"))?;

        mem_block.sub_allocator.free(chunk_id)?;

        if mem_block.sub_allocator.is_empty() {
            if mem_block.sub_allocator.supports_general_allocations() {
                if self.active_general_blocks > 1 {
                    let block = self.memory_blocks[block_idx].take();
                    let block =
                        block.ok_or(AllocationError::Internal("memory block must be Some"))?;
                    block.destroy(device);

                    self.active_general_blocks -= 1;
                }
            } else {
                let block = self.memory_blocks[block_idx].take();
                let block = block.ok_or(AllocationError::Internal("memory block must be Some"))?;
                block.destroy(device);
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
    ) -> Result<SubAllocation> {
        // vkGetBufferMemoryRequirements2()
        // VK_KHR_get_memory_requirements2
        self.allocate_memory(desc, true, false)
    }

    /// Allocates memory for an image.
    // TODO https://www.asawicki.info/articles/VK_KHR_dedicated_allocation.php5
    pub fn allocate_memory_for_image(
        &mut self,
        desc: &AllocationDescriptor,
    ) -> Result<SubAllocation> {
        // vkGetImageMemoryRequirements2()
        // VK_KHR_get_memory_requirements2
        // TODO how to decide if an image is linear?
        self.allocate_memory(desc, false, false)
    }

    fn allocate_memory(
        &mut self,
        desc: &AllocationDescriptor,
        linear: bool,
        dedicated: bool,
    ) -> Result<SubAllocation> {
        let size = desc.requirements.size;
        let alignment = desc.requirements.alignment;

        #[cfg(feature = "tracing")]
        debug!(
            "Allocating \"{}\" of {} bytes with an alignment of {}.",
            desc.name, size, alignment
        );

        if size == 0 || !alignment.is_power_of_two() {
            return Err(AllocationError::InvalidAllocationDescriptor);
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
            None => return Err(AllocationError::NoCompatibleMemoryTypeFound),
        };

        let sub_allocation = self.memory_types[memory_type_index].allocate(
            &self.device,
            desc,
            linear,
            dedicated,
            self.buffer_image_granularity,
        );

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
                    None => return Err(AllocationError::NoCompatibleMemoryTypeFound),
                };

                self.memory_types[memory_type_index].allocate(
                    &self.device,
                    desc,
                    linear,
                    dedicated,
                    self.buffer_image_granularity,
                )
            } else {
                sub_allocation
            }
        } else {
            sub_allocation
        }
    }

    /// Free memory of a sub allocation.
    pub fn free(&mut self, sub_allocation: SubAllocation) -> Result<()> {
        #[cfg(feature = "tracing")]
        debug!(
            "Freeing \"{}\".",
            sub_allocation.name.as_deref().unwrap_or("<null>")
        );

        if sub_allocation.is_null() {
            return Ok(());
        }

        self.memory_types[sub_allocation.memory_type_index].free(sub_allocation, &self.device)?;

        Ok(())
    }

    /// Logs memory leaks as warnings.
    #[cfg(feature = "tracing")]
    fn log_memory_leaks(&self) {
        for (mem_type_i, mem_type) in self.memory_types.iter().enumerate() {
            for (block_i, mem_block) in mem_type.memory_blocks.iter().enumerate() {
                if let Some(mem_block) = mem_block {
                    mem_block
                        .sub_allocator
                        .log_memory_leaks(mem_type_i, block_i);
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
                let block = mem_block.take();
                if let Some(block) = block {
                    block.destroy(&self.device);
                }
            }
        }
    }
}
