use anyhow::Result;
use ash::vk;

fn main() -> Result<()> {
    // Log level is based on RUST_LOG env var.
    tracing_subscriber::fmt::init();

    let instance = asche::Adapter::new(&asche::AdapterDescriptor {
        app_name: "".to_string(),
        app_version: ash::vk::make_version(0, 1, 0),
        vulkan_version: asche::VulkanVersion::V1,
    })?;
    let _device = instance.request_device(vk::PhysicalDeviceType::DISCRETE_GPU)?;
    Ok(())
}
