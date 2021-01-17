use anyhow::Result;
use ash::vk::PhysicalDeviceType;

use asche::{InstanceDescriptor, VulkanVersion};

fn main() -> Result<()> {
    // Log level is based on RUST_LOG env var.
    tracing_subscriber::fmt::init();

    let instance = asche::Instance::new(&InstanceDescriptor {
        app_name: "".to_string(),
        app_version: ash::vk::make_version(0, 1, 0),
        vulkan_version: VulkanVersion::V1,
    })?;
    let _device = instance.create_device(PhysicalDeviceType::DISCRETE_GPU)?;
    Ok(())
}
