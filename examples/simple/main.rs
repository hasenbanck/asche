use ash::vk;
use raw_window_handle::HasRawWindowHandle;

fn main() -> Result<(), asche::AscheError> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();

    // Log level is based on RUST_LOG env var.
    #[cfg(feature = "tracing")]
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let instance = asche::Instance::new(&asche::InstanceDescriptor {
        app_name: "simple example",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &window.raw_window_handle(),
    })?;

    let mut device = instance.request_device(&asche::DeviceDescriptor {
        ..Default::default()
    })?;

    device.recreate_swapchain(Some(vk::Extent2D {
        width: window.outer_size().width,
        height: window.outer_size().height,
    }))?;

    Ok(())
}
