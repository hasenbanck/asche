# asche

![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

Provides an abstraction layer above ash to easier use Vulkan in Rust with minimal dependencies.

This crate targets Vulkan 1.2 and uses the following extensions:

Vulan 1.1:

- VK_KHR_get_memory_requirements2
- VK_KHR_get_physical_device_properties2
- VK_KHR_maintenance3

Vulkan 1.2:

- VK_KHR_dedicated_allocation
- VK_KHR_buffer_device_address
- VK_KHR_imageless_framebuffer
- VK_KHR_timeline_semaphore
- VK_EXT_descriptor_indexing

## Status

Under heavy development. Not usable yet.

## Should you use this crate?

In 99.95% of the cases the answer to this question is "No".

* Do you want to target multiple modern graphic API with validation? Use wgpu.
* Do you want to target multiple modern graphic API without validation and a lot of pain? gfx-rs.
* Do you want to target Vulkan with validation? Use Vulkano.
* Do you want to target Vulkan, don't care for handholding and want a lot of pain? Use ash or erupt.
* Do you want to target Vulkan, want a little bit of handholding and a lot of pain? Use this crate.

## Credits

The device initialization code is a hard copy from the gfx-rs crate.

The current memory allocator is a hard copy from the gpu_allocator create.

## License

Licensed under MIT or Apache-2.0.
