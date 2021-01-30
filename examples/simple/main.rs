use anyhow::{anyhow, Result};
use raw_window_handle::HasRawWindowHandle;

fn main() -> Result<()> {
    let sdl_context = sdl2::init().map_err(|s| anyhow!(s))?;
    let video_subsystem = sdl_context.video().map_err(|s| anyhow!(s))?;

    let window = video_subsystem
        .window("simple example", 800, 600)
        .vulkan()
        .build()?;

    // Log level is based on RUST_LOG env var.
    tracing_subscriber::fmt::init();

    let instance = asche::Adapter::new(
        &window.raw_window_handle(),
        &asche::AdapterDescriptor {
            app_name: "".to_string(),
            app_version: ash::vk::make_version(0, 1, 0),
            vulkan_version: asche::VulkanVersion::V1,
        },
    )?;
    let _device = instance.request_device(&asche::DeviceDescriptor {
        ..Default::default()
    })?;
    Ok(())
}
