# asche

[![Latest version](https://img.shields.io/crates/v/asche.svg)](https://crates.io/crates/asche)
[![Documentation](https://docs.rs/vk-alloc/badge.svg)](https://docs.rs/asche)
![ZLIB](https://img.shields.io/badge/license-zlib-blue.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

Provides an abstraction layer above [erupt](https://crates.io/crates/erupt)
to easier use Vulkan in Rust. Mainly handles the Vulkan busywork (device initialization, memory
handling etc.) and the lifetimes of objects.

No validation and a lot of pain. You most likely want to use
[wgpu-rs](https://github.com/gfx-rs/wgpu-rs) instead!

## Requirements

Vulkan 1.2.

Used features:

- "buffer device address"
- "timeline_semaphores"

## Features

* `logging` Adds logging using [tracing](https://github.com/tokio-rs/tracing).
* `profiling` Adds support for [profiling](https://github.com/aclysma/profiling).
* `smallvec` Uses the crate [smallvec](https://github.com/servo/smallvec)
  for vectors in the hot path to reduce pressure on the heap.

`logging` and `smallvec` are on by default.

## Examples

Examples are provided.

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
