use ash::vk;
use raw_window_handle::HasRawWindowHandle;
use vk_shader_macros::include_glsl;

fn main() -> Result<(), asche::AscheError> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .with_title("asche - triangle example")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    // Log level is based on RUST_LOG env var.
    #[cfg(feature = "tracing")]
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let instance = asche::Instance::new(&asche::InstanceDescriptor {
        app_name: "triangle example",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &window.raw_window_handle(),
    })?;

    let (device, (_, graphics_queue, _)) = instance.request_device(&asche::DeviceDescriptor {
        ..Default::default()
    })?;

    let mut app = Application::new(device, graphics_queue, window)?;

    event_loop.run(move |event, _, control_flow| match event {
        winit::event::Event::WindowEvent {
            event: winit::event::WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = winit::event_loop::ControlFlow::Exit;
        }
        winit::event::Event::MainEventsCleared => {
            app.window.request_redraw();
        }
        winit::event::Event::RedrawRequested(_) => {
            app.render().unwrap();
        }
        _ => {}
    });
}

struct Application {
    device: asche::Device,
    graphics_queue: asche::GraphicsQueue,
    graphics_command_pool: asche::GraphicsCommandPool,
    window: winit::window::Window,
    extent: vk::Extent2D,
    graphics_pipeline: asche::GraphicsPipeline,
    render_pass: asche::RenderPass,
    frame_counter: u64,
}

impl Application {
    fn new(
        mut device: asche::Device,
        mut graphics_queue: asche::GraphicsQueue,
        window: winit::window::Window,
    ) -> Result<Self, asche::AscheError> {
        let extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        // Shader
        let vert_module = device.create_shader_module(
            "vertex module",
            include_glsl!("./examples/triangle/shaders/triangle.vert"),
        )?;
        let frag_module = device.create_shader_module(
            "fragment module",
            include_glsl!("./examples/triangle/shaders/triangle.frag"),
        )?;

        let mainfunctionname = std::ffi::CString::new("main").unwrap();
        let vertexshader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module.raw)
            .name(&mainfunctionname);
        let fragmentshader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module.raw)
            .name(&mainfunctionname);

        // Renderpass
        let attachments = [vk::AttachmentDescription::builder()
            .format(vk::Format::B8G8R8A8_SRGB)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .samples(vk::SampleCountFlags::TYPE_1)
            .build()];

        let color_attachment_references = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_references)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build()];

        let subpass_dependencies = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_subpass(0)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build()];

        let renderpass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&subpass_dependencies);

        let render_pass =
            device.create_render_pass("Graphics Render Pass Simple", renderpass_info)?;

        // Pipeline layout
        let pipeline_layout = vk::PipelineLayoutCreateInfo::builder();
        let pipeline_layout =
            device.create_pipeline_layout("Pipeline Layout Simple", pipeline_layout)?;

        // Pipeline
        let shader_stages = vec![vertexshader_stage.build(), fragmentshader_stage.build()];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .scissor_count(1)
            .viewport_count(1);
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::NONE)
            .polygon_mode(vk::PolygonMode::FILL);
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
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
            )
            .build()];
        let color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
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
        let pipeline = device.create_graphics_pipeline("triangle pipeline", pipeline_info)?;

        let graphics_command_pool = graphics_queue.create_command_pool()?;

        Ok(Self {
            device,
            graphics_queue,
            graphics_command_pool,
            window,
            extent,
            graphics_pipeline: pipeline,
            render_pass,
            frame_counter: 0,
        })
    }

    fn render(&mut self) -> Result<(), asche::AscheError> {
        let frame_offset = self.frame_counter * Timeline::RenderEnd as u64;
        let frame = self.device.get_next_frame()?;

        let graphics_buffer = self.graphics_command_pool.create_command_buffer(
            Timeline::RenderStart.with_offset(frame_offset),
            Timeline::RenderEnd.with_offset(frame_offset),
        )?;

        graphics_buffer.record(|encoder| {
            encoder.set_viewport_and_scissor(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            });

            let frame_buffer = self.device.get_frame_buffer(&self.render_pass, &frame)?;

            let pass = encoder.begin_render_pass(
                &self.render_pass,
                frame_buffer,
                &[vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [1.0, 0.0, 1.0, 1.0],
                    },
                }],
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                },
            )?;

            pass.cmd_bind_pipeline(&self.graphics_pipeline);
            pass.cmd_draw(3, 1, 0, 0);

            Ok(())
        })?;

        self.graphics_queue.execute(&graphics_buffer)?;
        self.graphics_queue
            .wait_for_timeline_value(Timeline::RenderEnd.with_offset(frame_offset))?;

        self.graphics_command_pool.reset()?;

        self.device.queue_frame(&self.graphics_queue, frame)?;
        self.frame_counter += 1;

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
