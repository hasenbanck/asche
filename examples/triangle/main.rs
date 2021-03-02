use erupt::vk;

fn main() -> Result<(), asche::AscheError> {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem
        .window("asche - triangle example", 1920, 1080)
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

    let instance = asche::Instance::new(
        &window,
        asche::InstanceConfiguration {
            app_name: "triangle example",
            app_version: erupt::vk::make_version(1, 0, 0),
            extensions: vec![],
        },
    )?;

    let (device, (_, graphics_queue, _)) = instance.request_device(asche::DeviceConfiguration {
        ..Default::default()
    })?;

    let mut app = Application::new(device, graphics_queue, &window)?;

    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. }
                | sdl2::event::Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        app.render()?;
    }

    Ok(())
}

struct Application {
    device: asche::Device,
    graphics_queue: asche::GraphicsQueue,
    graphics_command_pool: asche::GraphicsCommandPool,
    extent: vk::Extent2D,
    graphics_pipeline: asche::GraphicsPipeline,
    render_pass: asche::RenderPass,
    timeline: asche::TimelineSemaphore,
    timeline_value: u64,
}

impl Application {
    fn new(
        mut device: asche::Device,
        mut graphics_queue: asche::GraphicsQueue,
        window: &sdl2::video::Window,
    ) -> Result<Self, asche::AscheError> {
        let (width, height) = window.size();
        let extent = vk::Extent2D { width, height };

        // Shader
        let vert_module = device.create_shader_module(
            "Vertex Shader Module",
            include_bytes!("../../gen/shader/triangle.vert.spv"),
        )?;
        let frag_module = device.create_shader_module(
            "Fragment Shader Module",
            include_bytes!("../../gen/shader/triangle.frag.spv"),
        )?;

        let mainfunctionname = std::ffi::CString::new("main").unwrap();
        let vertexshader_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(vert_module.raw)
            .name(&mainfunctionname);
        let fragmentshader_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(frag_module.raw)
            .name(&mainfunctionname);

        // Renderpass
        let attachments = [vk::AttachmentDescriptionBuilder::new()
            .format(vk::Format::B8G8R8A8_SRGB)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .samples(vk::SampleCountFlagBits::_1)];

        let color_attachment_references = [vk::AttachmentReferenceBuilder::new()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let subpasses = [vk::SubpassDescriptionBuilder::new()
            .color_attachments(&color_attachment_references)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];

        let subpass_dependencies = [vk::SubpassDependencyBuilder::new()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_subpass(0)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )];

        let renderpass_info = vk::RenderPassCreateInfoBuilder::new()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&subpass_dependencies);

        let render_pass =
            device.create_render_pass("Graphics Render Pass Simple", renderpass_info)?;

        // Pipeline layout
        let pipeline_layout = vk::PipelineLayoutCreateInfoBuilder::new();
        let pipeline_layout =
            device.create_pipeline_layout("Pipeline Layout Simple", pipeline_layout)?;

        // Pipeline
        let shader_stages = vec![vertexshader_stage, fragmentshader_stage];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfoBuilder::new();
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let dynamic_state = vk::PipelineDynamicStateCreateInfoBuilder::new()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
            .scissor_count(1)
            .viewport_count(1);
        let rasterization_state = vk::PipelineRasterizationStateCreateInfoBuilder::new()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::NONE)
            .polygon_mode(vk::PolygonMode::FILL);
        let multisample_state = vk::PipelineMultisampleStateCreateInfoBuilder::new()
            .rasterization_samples(vk::SampleCountFlagBits::_1);
        let color_blend_attachments = [vk::PipelineColorBlendAttachmentStateBuilder::new()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfoBuilder::new()
            .attachments(&color_blend_attachments);

        let pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .dynamic_state(&dynamic_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(pipeline_layout.raw)
            .render_pass(render_pass.raw)
            .subpass(0);
        let pipeline = device.create_graphics_pipeline("Triangle Pipeline", pipeline_info)?;

        let graphics_command_pool = graphics_queue.create_command_pool()?;

        let timeline_value = 0;
        let timeline = device.create_timeline_semaphore("Render Timeline", timeline_value)?;

        Ok(Self {
            device,
            graphics_queue,
            graphics_command_pool,
            extent,
            graphics_pipeline: pipeline,
            render_pass,
            timeline,
            timeline_value,
        })
    }

    fn render(&mut self) -> Result<(), asche::AscheError> {
        let frame = self.device.get_next_frame()?;

        let graphics_buffer = self.graphics_command_pool.create_command_buffer(
            &self.timeline,
            Timeline::RenderStart.with_offset(self.timeline_value),
            Timeline::RenderEnd.with_offset(self.timeline_value),
        )?;

        graphics_buffer.record(|encoder| {
            encoder.set_viewport_and_scissor(
                vk::Rect2DBuilder::new()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(self.extent),
            );

            let pass = encoder.begin_render_pass(
                &self.render_pass,
                &[&asche::RenderPassColorAttachmentDescriptor {
                    attachment: frame.view,
                    clear_value: vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [1.0, 0.0, 1.0, 1.0],
                        },
                    },
                }],
                None,
                self.extent,
            )?;

            pass.bind_pipeline(&self.graphics_pipeline);
            pass.draw(3, 1, 0, 0);

            Ok(())
        })?;

        self.graphics_queue.submit(&graphics_buffer)?;
        self.timeline
            .wait_for_value(Timeline::RenderEnd.with_offset(self.timeline_value))?;

        self.graphics_command_pool.reset()?;

        self.device.queue_frame(&self.graphics_queue, frame)?;
        self.timeline_value += Timeline::RenderEnd as u64;

        Ok(())
    }
}

#[derive(Copy, Clone)]
enum Timeline {
    RenderStart = 0,
    RenderEnd,
}

impl Timeline {
    fn with_offset(&self, offset: u64) -> u64 {
        *self as u64 + offset
    }
}
