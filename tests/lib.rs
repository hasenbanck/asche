use raw_window_handle::HasRawWindowHandle;

use fixture::TestContext;

mod fixture;

#[test]
fn context_creation() {
    let test_context = TestContext::default();

    let _ = asche::Instance::new(&asche::InstanceDescriptor {
        app_name: "context_creation",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &test_context.window.raw_window_handle(),
    });
}

#[test]
fn device_creation() {
    let test_context = TestContext::default();

    let instance = asche::Instance::new(&asche::InstanceDescriptor {
        app_name: "device_creation",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &test_context.window.raw_window_handle(),
    })
    .unwrap();

    let _device = instance
        .request_device(&asche::DeviceDescriptor {
            ..Default::default()
        })
        .unwrap();
}
