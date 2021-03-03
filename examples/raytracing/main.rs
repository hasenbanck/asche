use bytemuck::{Pod, Zeroable};
use erupt::vk;
use ultraviolet::{Vec2, Vec4};

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: Vec4,
    tex_coord: Vec2,
}

unsafe impl Pod for Vertex {}

unsafe impl Zeroable for Vertex {}

fn main() -> Result<(), asche::AscheError> {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem
        .window("asche - raytracing example", 1920, 1080)
        .vulkan()
        .allow_highdpi()
        .build()
        .unwrap();

    // Log level is based on RUST_LOG env var.
    #[cfg(feature = "tracing")]
    {
        let filter = tracing_subscriber::EnvFilter::from_default_env();
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    let instance = asche::Instance::new(
        &window,
        asche::InstanceConfiguration {
            app_name: "raytracing example",
            app_version: erupt::vk::make_version(1, 0, 0),
            extensions: vec![],
        },
    )?;

    let (_, (_, _, _)) = instance.request_device(asche::DeviceConfiguration {
        extensions: vec![
            vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            vk::KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
            vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        ],
        features_v1_0: Some(
            vk::PhysicalDeviceFeaturesBuilder::new()
                .sampler_anisotropy(true)
                .texture_compression_bc(true),
        ),
        features_v1_2: Some(
            vk::PhysicalDeviceVulkan12FeaturesBuilder::new()
                .timeline_semaphore(true)
                .buffer_device_address(true)
                .scalar_block_layout(true)
                .descriptor_indexing(true)
                .uniform_buffer_standard_layout(true),
        ),
        features_raytracing: Some(
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHRBuilder::new()
                .ray_tracing_pipeline(true),
        ),
        features_acceleration_structure: Some(
            vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder::new()
                .acceleration_structure(true),
        ),
        ..Default::default()
    })?;

    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. }
                | sdl2::event::Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }
    }

    Ok(())
}

struct RaytracingApplication {
    renderpass: asche::RenderPass,
    offscreen_pipeline: asche::GraphicsPipeline,
    raytracing_pipeline: asche::RayTracingPipeline,
}
