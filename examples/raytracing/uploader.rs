use erupt::vk;

use asche::{CommandBufferSemaphore, CommonCommands};

use crate::{Lifetime, Result};

pub struct Uploader {
    staging_buffer: asche::Buffer<Lifetime>,
    timeline: asche::TimelineSemaphore,
    timeline_value: u64,
    transfer_pool: asche::TransferCommandPool,
    transfer_queue: asche::TransferQueue,
}

impl Uploader {
    pub unsafe fn new(
        device: &asche::Device<Lifetime>,
        mut transfer_queue: asche::TransferQueue,
    ) -> Result<Self> {
        let timeline_value = 0;
        let timeline = device.create_timeline_semaphore("Upload Timeline", timeline_value)?;
        let transfer_pool = transfer_queue.create_command_pool()?;

        let staging_buffer = device.create_buffer(&asche::BufferDescriptor::<_> {
            name: "Staging Buffer",
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_location: vk_alloc::MemoryLocation::CpuToGpu,
            lifetime: Lifetime::Buffer,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            size: 1024 * 1024, // Fixed 1 MB
            flags: None,
        })?;

        Ok(Self {
            staging_buffer,
            timeline,
            timeline_value,
            transfer_pool,
            transfer_queue,
        })
    }

    pub unsafe fn create_buffer_with_data(
        &mut self,
        device: &asche::Device<Lifetime>,
        name: &str,
        buffer_data: &[u8],
        buffer_type: vk::BufferUsageFlags,
        queues: vk::QueueFlags,
    ) -> Result<asche::Buffer<Lifetime>> {
        let data_size = buffer_data.len();
        let staging_slice = self
            .staging_buffer
            .mapped_slice_mut()?
            .expect("staging buffer allocation could not be not mapped");
        staging_slice[..data_size].clone_from_slice(buffer_data);

        let dst_buffer = device.create_buffer(&asche::BufferDescriptor::<_> {
            name,
            usage: buffer_type | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            lifetime: Lifetime::Buffer,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::TRANSFER | queues,
            size: data_size as u64,
            flags: None,
        })?;

        self.timeline_value += 1;
        let transfer_buffer = self.transfer_pool.create_command_buffer(
            &[],
            &[CommandBufferSemaphore::Timeline {
                semaphore: self.timeline.handle(),
                stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                value: self.timeline_value,
            }],
        )?;

        {
            let encoder = transfer_buffer.record()?;
            encoder.copy_buffer(
                self.staging_buffer.raw(),
                dst_buffer.raw(),
                0,
                0,
                buffer_data.len() as u64,
            );
        }

        self.transfer_queue.submit(&transfer_buffer, None)?;
        self.timeline.wait_for_value(self.timeline_value)?;

        self.transfer_pool.reset()?;

        Ok(dst_buffer)
    }
}
