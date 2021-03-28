use erupt::vk;

use asche::{QueueConfiguration, Queues};

fn main() -> Result<(), asche::AscheError> {
    // Asche doesn't support headless compute only setups!
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("asche - compute example")
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .build(&event_loop)
        .unwrap();

    // Log level is based on RUST_LOG env var.
    #[cfg(feature = "tracing")]
    {
        let filter = tracing_subscriber::EnvFilter::from_default_env();
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    let instance = asche::Instance::new(
        &window,
        asche::InstanceConfiguration {
            app_name: "compute example",
            app_version: erupt::vk::make_version(1, 0, 0),
            extensions: vec![],
        },
    )?;

    let (device, queues) = instance.request_device(asche::DeviceConfiguration {
        queue_configuration: QueueConfiguration {
            compute_queues: vec![1.0],
            graphics_queues: vec![],
            transfer_queues: vec![],
        },
        ..Default::default()
    })?;

    let Queues {
        mut compute_queues,
        graphics_queues: _graphics_queues,
        transfer_queues: _transfer_queues,
    } = queues;

    let mut app = Application::new(device, compute_queues.pop().unwrap())?;
    app.compute()?;

    Ok(())
}

struct Application {
    device: asche::Device,
    compute_queue: asche::ComputeQueue,
    compute_command_pool: asche::ComputeCommandPool,
    pipeline: asche::ComputePipeline,
    pipeline_layout: asche::PipelineLayout,
    timeline: asche::TimelineSemaphore,
    timeline_value: u64,
    descriptor_set_layout: asche::DescriptorSetLayout,
    descriptor_pool: asche::DescriptorPool,
}

impl Application {
    fn new(
        device: asche::Device,
        mut compute_queue: asche::ComputeQueue,
    ) -> Result<Self, asche::AscheError> {
        // Shader
        let comp_module = device.create_shader_module(
            "Compute Shader Module",
            include_bytes!("shader/compute.comp.spv"),
        )?;

        let mainfunctionname = std::ffi::CString::new("main").unwrap();

        let compute_shader_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(comp_module.raw)
            .name(&mainfunctionname);

        // Descriptor set layout
        let bindings = [vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);
        let descriptor_set_layout =
            device.create_descriptor_set_layout("Compute Descriptor Set Layout", layout_info)?;

        // Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSizeBuilder::new()
            .descriptor_count(1)
            ._type(vk::DescriptorType::STORAGE_BUFFER)];

        let descriptor_pool = device.create_descriptor_pool(&asche::DescriptorPoolDescriptor {
            name: "Compute Descriptor Pool",
            max_sets: 16,
            pool_sizes: &pool_sizes,
            flags: None,
        })?;

        // Pipeline layout
        let layouts = [descriptor_set_layout.raw];
        let pipeline_layout = vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&layouts);
        let pipeline_layout =
            device.create_pipeline_layout("Compute Pipeline Layout", pipeline_layout)?;

        // Pipeline
        let pipeline_info = vk::ComputePipelineCreateInfoBuilder::new()
            .layout(pipeline_layout.raw)
            .stage(compute_shader_stage.build());

        let pipeline = device.create_compute_pipeline("Compute Pipeline", pipeline_info)?;

        let compute_command_pool = compute_queue.create_command_pool()?;

        let timeline_value = 0;
        let timeline = device.create_timeline_semaphore("Compute Timeline", timeline_value)?;

        Ok(Self {
            device,
            compute_queue,
            compute_command_pool,
            pipeline,
            pipeline_layout,
            timeline,
            timeline_value,
            descriptor_set_layout,
            descriptor_pool,
        })
    }

    fn compute(&mut self) -> Result<(), asche::AscheError> {
        const ELEMENTS: usize = 64 * 1024;
        const DATA_SIZE: u64 = (ELEMENTS * std::mem::size_of::<u32>()) as u64;

        let mut data: Vec<u32> = vec![0; ELEMENTS];
        data.iter_mut()
            .enumerate()
            .for_each(|(id, x)| *x = id as u32);

        let mut buffer = self.device.create_buffer(&asche::BufferDescriptor {
            name: "Input Buffer",
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_location: vk_alloc::MemoryLocation::CpuToGpu,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queues: vk::QueueFlags::COMPUTE,
            size: DATA_SIZE,
            flags: None,
        })?;

        {
            let data_slice = buffer
                .allocation
                .mapped_slice_mut()
                .expect("data buffer allocation was not mapped");
            data_slice[..].clone_from_slice(bytemuck::cast_slice(&data));
            self.device.flush_mapped_memory(&buffer.allocation)?;
        }

        let compute_buffer = self.compute_command_pool.create_command_buffer(
            &self.timeline,
            self.timeline_value,
            self.timeline_value + 1,
        )?;

        let set = self.descriptor_pool.create_descriptor_set(
            "Compute Descriptor Set",
            &self.descriptor_set_layout,
            None,
        )?;

        let buffer_info = [vk::DescriptorBufferInfoBuilder::new()
            .buffer(buffer.raw)
            .offset(0)
            .range(DATA_SIZE)];
        let write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(set.raw)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info);
        self.device.update_descriptor_sets(&[write], &[]);

        {
            let encoder = compute_buffer.record()?;
            encoder.bind_pipeline(&self.pipeline);
            encoder.bind_descriptor_sets(self.pipeline_layout.raw, 0, &[set.raw], &[]);
            encoder.dispatch(1024, 1, 1);
        }
        self.compute_queue.submit(&compute_buffer)?;
        self.timeline_value += 1;
        self.timeline.wait_for_value(self.timeline_value)?;

        {
            let data_slice = buffer
                .allocation
                .mapped_slice()
                .expect("data buffer allocation was not mapped");
            data[..].clone_from_slice(bytemuck::cast_slice(&data_slice));
        }

        data.iter()
            .enumerate()
            .for_each(|(id, output)| assert_eq!((id * 42) as u32, *output));
        Ok(())
    }
}
