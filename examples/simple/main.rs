use anyhow::Result;
use raw_window_handle::HasRawWindowHandle;

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    // Log level is based on RUST_LOG env var.
    #[cfg(feature = "tracing")]
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let context = asche::Context::new(&asche::ContextDescriptor {
        app_name: "simple example",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &window.raw_window_handle(),
    })?;

    let _device = asche::Device::new(
        context,
        &asche::DeviceDescriptor {
            ..Default::default()
        },
    )?;

    Ok(())
}
