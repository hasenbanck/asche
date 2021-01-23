use anyhow::Result;
use raw_window_handle::HasRawWindowHandle;

use asche::DeviceDescriptor;

fn main() -> Result<()> {
    let eventloop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&eventloop)?;

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
    let _device = instance.request_device(&DeviceDescriptor {
        ..Default::default()
    })?;
    Ok(())
}
