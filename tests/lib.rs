mod fixture;

use fixture::TestContext;
use raw_window_handle::HasRawWindowHandle;

#[test]
fn context_creation() {
    let test_context = TestContext::default();

    let _ = asche::Context::new(&asche::ContextDescriptor {
        app_name: "context_creation",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &test_context.window.raw_window_handle(),
    });
}

#[test]
fn device_creation() {
    let test_context = TestContext::default();

    let context = asche::Context::new(&asche::ContextDescriptor {
        app_name: "device_creation",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &test_context.window.raw_window_handle(),
    })
    .unwrap();

    let _ = asche::Device::new(
        context,
        &asche::DeviceDescriptor {
            ..Default::default()
        },
    )
    .unwrap();
}
