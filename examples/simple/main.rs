use anyhow::Result;
use raw_window_handle::HasRawWindowHandle;

fn main() -> Result<()> {
    let eventloop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&eventloop)?;

    // Log level is based on RUST_LOG env var.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let adapter = asche::Adapter::new(
        &window.raw_window_handle(),
        &asche::AdapterDescriptor {
            app_name: "".to_string(),
            app_version: ash::vk::make_version(0, 1, 0),
            ..Default::default()
        },
    )?;
    let _device = adapter.request_device(&asche::DeviceDescriptor {
        ..Default::default()
    })?;

    Ok(())
}
