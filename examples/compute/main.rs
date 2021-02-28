use ash::vk;
use raw_window_handle::HasRawWindowHandle;

fn main() -> Result<(), asche::AscheError> {
    // Asche doesn't support headless compute only setups!
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem
        .window("asche - cube example", 800, 600)
        .vulkan()
        .allow_highdpi()
        .build()
        .unwrap();

    // Log level is based on RUST_LOG env var.
    #[cfg(feature = "tracing")]
    {
        let filter = tracing_subscriber::EnvFilter::from_default_env();
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    let instance = asche::Instance::new(asche::InstanceConfiguration {
        app_name: "compute example",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &window.raw_window_handle(),
        extensions: vec![],
    })?;

    let (device, (compute_queue, _, _)) = instance.request_device(asche::DeviceConfiguration {
        ..Default::default()
    })?;

    let mut app = Application::new(device, compute_queue)?;
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
        mut device: asche::Device,
        mut compute_queue: asche::ComputeQueue,
    ) -> Result<Self, asche::AscheError> {
        // Shader
        let comp_module = device.create_shader_module(
            "Compute Shader Module",
            include_bytes!("../../gen/shader/compute.comp.spv"),
        )?;

        let mainfunctionname = std::ffi::CString::new("main").unwrap();

        let compute_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(comp_module.raw)
            .name(&mainfunctionname);

        // Descriptor set layout
        let bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build()];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let descriptor_set_layout =
            device.create_descriptor_set_layout("Compute Descriptor Set Layout", layout_info)?;

        // Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize::builder()
            .descriptor_count(1)
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .build()];

        let descriptor_pool = device.create_descriptor_pool(&asche::DescriptorPoolDescriptor {
            name: "Compute Descriptor Pool",
            max_sets: 16,
            pool_sizes: &pool_sizes,
            flags: None,
        })?;

        // Pipeline layout
        let layouts = [descriptor_set_layout.raw];
        let pipeline_layout = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
        let pipeline_layout =
            device.create_pipeline_layout("Compute Pipeline Layout", pipeline_layout)?;

        // Pipeline
        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
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

        let set = self
            .descriptor_pool
            .create_descriptor_set("Compute Descriptor Set", &self.descriptor_set_layout)?;

        set.update(&asche::UpdateDescriptorSetDescriptor {
            binding: 0,
            update: asche::DescriptorSetUpdate::StorageBuffer {
                buffer: &buffer,
                offset: 0,
                range: DATA_SIZE,
            },
        });

        compute_buffer.record(|encoder| {
            encoder.bind_pipeline(&self.pipeline);
            encoder.bind_descriptor_set(&self.pipeline_layout, 0, &set, &[]);
            encoder.dispatch(1024, 1, 1);
            Ok(())
        })?;

        self.compute_queue.execute(&compute_buffer)?;
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
