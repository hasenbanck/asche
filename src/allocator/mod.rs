//! This module is a hard copy of the [gpu-allocator](https://github.com/Traverse-Research/gpu-allocator) project.
//!
//! ## Setting up the memory allocator
//!
//! ```rust
//! use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
//! use ash::vk;
//! use asche::allocator::{Allocator, AllocatorDescriptor};
//!
//! let mut allocator = Allocator::new(&AllocatorDescriptor {
//!     instance,
//!     logical_device:device,
//!     physical_device,
//! });
//! ```
//!
//! ## Simple allocation example
//!
//! ```rust
//! use ash::vk;
//! use asche::allocator::{AllocationDescriptor, MemoryLocation};
//!
//! // Setup vulkan info
//! let vk_info = vk::BufferCreateInfo::builder()
//!     .size(512)
//!     .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
//!
//! let buffer = unsafe { device.create_buffer(&vk_info, None) }?;
//! let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
//!
//! let allocation = allocator
//!     .allocate(&AllocationDescriptor {
//!         name: "Example allocation",
//!         requirements,
//!         location: MemoryLocation::CpuToGpu,
//!         linear: true,
//!     })?;
//!
//! unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };
//!
//! allocator.free(allocation)?;
//! unsafe { device.destroy_buffer(buffer, None) };
//! ```
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;
#[cfg(feature = "tracing")]
use tracing::{debug, trace, warn};

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
    /// If the resource is linear (buffer / linear texture) or a regular (tiled) texture.
    pub linear: bool,
}

/// The location of the memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLocation {
    /// The allocated resource is stored at an unknown memory location. Let the driver decide what's the best location.
    Unknown,
    /// Store the allocation in GPU only accessible memory. Typically this is the faster GPU resource and this should be
    /// where most of the allocations live.
    GpuOnly,
    /// Memory useful for uploading data to the GPU and potentially for constant buffers.
    CpuToGpu,
    /// Memory useful for reading back data from GPU to CPU.
    GpuToCpu,
}

/// The configuration descriptor for the allocator.
pub struct AllocatorDescriptor {
    /// The vulkan instance which will use this allocator.
    pub instance: ash::Instance,
    /// The logical device which will use this allocator.
    pub logical_device: ash::Device,
    /// The physical device which will use this allocator.
    pub physical_device: ash::vk::PhysicalDevice,
}

trait SubAllocator: std::fmt::Debug {
    fn allocate(
        &mut self,
        size: u64,
        alignment: u64,
        allocation_type: AllocationType,
        granularity: u64,
        name: &'static str,
    ) -> Result<(u64, std::num::NonZeroU64)>;

    fn free(&mut self, sub_allocation: SubAllocation) -> Result<()>;

    /// Logs memory leaks as warnings.
    fn log_memory_leaks(&self, memory_type_index: usize, memory_block_index: usize);

    #[must_use]
    fn supports_general_allocations(&self) -> bool;

    #[must_use]
    fn size(&self) -> u64;

    #[must_use]
    fn allocated(&self) -> u64;

    /// Reports how much memory is available in this sub allocator
    #[must_use]
    fn available_memory(&self) -> u64 {
        self.size() - self.allocated()
    }

    /// Reports if the sub allocator is empty (having no allocations).
    #[must_use]
    fn is_empty(&self) -> bool {
        self.allocated() == 0
    }
}

/// A sub allocation.
#[derive(Clone, Debug)]
pub struct SubAllocation {
    chunk_id: Option<std::num::NonZeroU64>,
    memory_block_index: usize,
    memory_type_index: usize,
    device_memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    mapped_ptr: Option<std::ptr::NonNull<std::ffi::c_void>>,
    name: Option<&'static str>,
}

unsafe impl Send for SubAllocation {}

impl SubAllocation {
    /// Returns the `vk::DeviceMemory` object that is backing this allocation.
    /// This memory object can be shared with multiple other allocations and shouldn't be free'd (or allocated from)
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

#[derive(PartialEq, Copy, Clone, Debug)]
#[repr(u8)]
pub(crate) enum AllocationType {
    Free,
    Linear,
    NonLinear,
}

#[derive(Debug)]
struct MemoryBlock {
    device_memory: vk::DeviceMemory,
    size: u64,
    mapped_ptr: *mut std::ffi::c_void,
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
            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(size)
                .memory_type_index(mem_type_index as u32);

            let allocation_flags = vk::MemoryAllocateFlags::DEVICE_ADDRESS;
            let mut flags_info = vk::MemoryAllocateFlagsInfo::builder().flags(allocation_flags);

            // FIXME Activate the needed extension for this.
            let alloc_info = if cfg!(feature = "vulkan_device_address") {
                alloc_info.push_next(&mut flags_info)
            } else {
                alloc_info
            };

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

const DEFAULT_DEVICE_MEMORY_BLOCK_SIZE: u64 = 256 * 1024 * 1024;
const DEFAULT_HOST_MEMORY_BLOCK_SIZE: u64 = 64 * 1024 * 1024;

impl MemoryType {
    fn allocate(
        &mut self,
        device: &ash::Device,
        desc: &AllocationDescriptor,
        granularity: u64,
    ) -> Result<SubAllocation> {
        let allocation_type = if desc.linear {
            AllocationType::Linear
        } else {
            AllocationType::NonLinear
        };

        let memory_block_size = if self
            .memory_properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            DEFAULT_HOST_MEMORY_BLOCK_SIZE
        } else {
            DEFAULT_DEVICE_MEMORY_BLOCK_SIZE
        };

        let size = desc.requirements.size;
        let alignment = desc.requirements.alignment;

        // Create a dedicated block for large memory allocations
        if size > memory_block_size {
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
                        let mapped_ptr = if !mem_block.mapped_ptr.is_null() {
                            let offset_ptr = unsafe { mem_block.mapped_ptr.add(offset as usize) };
                            std::ptr::NonNull::new(offset_ptr)
                        } else {
                            None
                        };
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
                }
            }
        };

        let mapped_ptr = if !mem_block.mapped_ptr.is_null() {
            let offset_ptr = unsafe { mem_block.mapped_ptr.add(offset as usize) };
            std::ptr::NonNull::new(offset_ptr)
        } else {
            None
        };

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

        mem_block.sub_allocator.free(sub_allocation)?;

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
    physical_mem_props: vk::PhysicalDeviceMemoryProperties,
    buffer_image_granularity: u64,
}

impl Allocator {
    pub fn new(desc: &AllocatorDescriptor) -> Self {
        let mem_props = unsafe {
            desc.instance
                .get_physical_device_memory_properties(desc.physical_device)
        };

        let memory_types = &mem_props.memory_types[..mem_props.memory_type_count as _];

        #[cfg(feature = "tracing")]
        {
            trace!("Memory type count: {}", mem_props.memory_type_count);
            trace!("Memory heap count: {}", mem_props.memory_heap_count);

            for (i, mem_type) in memory_types.iter().enumerate() {
                let flags = mem_type.property_flags;
                trace!(
                    "Memory type[{}]: prop flags: 0x{:x}, heap[{}]",
                    i,
                    flags.as_raw(),
                    mem_type.heap_index,
                );
            }
            for i in 0..mem_props.memory_heap_count {
                trace!(
                    "Heap[{}] flags: 0x{:x}, size: {} MiB",
                    i,
                    mem_props.memory_heaps[i as usize].flags.as_raw(),
                    mem_props.memory_heaps[i as usize].size / (1024 * 1024)
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

        let physical_device_properties = unsafe {
            desc.instance
                .get_physical_device_properties(desc.physical_device)
        };

        let granularity = physical_device_properties.limits.buffer_image_granularity;

        Self {
            memory_types,
            device: desc.logical_device.clone(),
            physical_mem_props: mem_props,
            buffer_image_granularity: granularity,
        }
    }

    pub fn allocate(&mut self, desc: &AllocationDescriptor) -> Result<SubAllocation> {
        let size = desc.requirements.size;
        let alignment = desc.requirements.alignment;

        #[cfg(feature = "tracing")]
        debug!(
            "Allocating \"{}\" of {} bytes with an alignment of {}.",
            desc.name, size, alignment
        );

        if size == 0 || !alignment.is_power_of_two() {
            return Err(AllocationError::InvalidAllocationCreateDesc);
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
            MemoryLocation::Unknown => vk::MemoryPropertyFlags::empty(),
        };
        let mut memory_type_index_opt = find_memory_type_index(
            &desc.requirements,
            &self.physical_mem_props,
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
                MemoryLocation::Unknown => vk::MemoryPropertyFlags::empty(),
            };

            memory_type_index_opt = find_memory_type_index(
                &desc.requirements,
                &self.physical_mem_props,
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
            self.buffer_image_granularity,
        );

        if desc.location == MemoryLocation::CpuToGpu {
            if sub_allocation.is_err() {
                let mem_loc_preferred_bits =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

                let memory_type_index_opt = find_memory_type_index(
                    &desc.requirements,
                    &self.physical_mem_props,
                    mem_loc_preferred_bits,
                );

                let memory_type_index = match memory_type_index_opt {
                    Some(x) => x as usize,
                    None => return Err(AllocationError::NoCompatibleMemoryTypeFound),
                };

                self.memory_types[memory_type_index].allocate(
                    &self.device,
                    desc,
                    self.buffer_image_granularity,
                )
            } else {
                sub_allocation
            }
        } else {
            sub_allocation
        }
    }

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

    #[cfg(feature = "tracing")]
    pub fn report_memory_leaks(&self) {
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
        self.report_memory_leaks();

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
