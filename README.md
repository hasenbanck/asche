# asche

![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

Provides an abstraction layer above [erupt](https://crates.io/crates/erupt)
to easier use Vulkan in Rust. Mainly handles the Vulkan busywork (device initialization, memory
handling etc.) and the lifetimes of objects.

No validation and a lot of pain.

## Requirements

Vulkan 1.2.

Used features:

- "buffer device address"
- "timeline_semaphores"

## Status

General API is finished. Not stability

## Examples

Examples are provided. They use SDL2 for windowing.

### Triangle

Most simple example that shows how to draw a triangle in Vulkan.

![Triangle example](assets/triangle.jpg)

### Cube

Shows how to use push constants, vertex and index buffers and also compressed textures.

![Cube example](assets/cube.jpg)

### Raytracing

Uses the `VK_raytracing_KHR` extension to fully ray trace a simple scene. Shows how to initialize
and use acceleration structures (triangle based), create and use the shader binding table (SBT), use
descriptor indexing with non uniform indexes and partial binds, write raytracing shader and do very
basic lightning.

![Raytracing example](assets/raytracing.jpg)

### Compute

Simple compute example.

## License

Licensed under MIT or Apache-2.0 or ZLIB.
