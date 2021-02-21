use ash::vk;
use raw_window_handle::HasRawWindowHandle;
use vk_shader_macros::include_glsl;

fn main() -> Result<(), asche::AscheError> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .with_title("asche - simple example")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

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

    let device = instance.request_device(&asche::DeviceDescriptor {
        ..Default::default()
    })?;

    let app = Application::new(device, window)?;

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
    window: winit::window::Window,
    extent: vk::Extent2D,
    _pipeline: asche::Pipeline,
    _command_pool: asche::CommandPool,
    command_buffer: asche::CommandBuffer,
    render_pass: asche::RenderPass,
}

impl Application {
    fn new(
        mut device: asche::Device,
        window: winit::window::Window,
    ) -> Result<Self, asche::AscheError> {
        let extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        // Shader
        let vert_module = device.create_shader_module(
            "vertex module",
            include_glsl!("./examples/simple/shaders/simple.vert"),
        )?;
        let frag_module = device.create_shader_module(
            "fragment module",
            include_glsl!("./examples/simple/shaders/simple.frag"),
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

        let render_pass = device.create_render_pass("simple render pass", renderpass_info)?;

        // Pipeline layout
        let pipeline_layout = vk::PipelineLayoutCreateInfo::builder();
        let pipeline_layout =
            device.create_pipeline_layout("simple pipeline layout", pipeline_layout)?;

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
        let pipeline = device.create_graphics_pipeline("simple pipeline", pipeline_info)?;

        let mut command_pool = device.create_command_pool(asche::QueueType::Graphics)?;
        let command_buffer = command_pool.create_command_buffer()?;

        Ok(Self {
            device,
            window,
            extent,
            _pipeline: pipeline,
            _command_pool: command_pool,
            command_buffer,
            render_pass,
        })
    }

    fn render(&self) -> Result<(), asche::AscheError> {
        let frame = self.device.get_next_frame()?;

        self.command_buffer.record(|encoder| {
            encoder.set_viewport_and_scissor(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            });

            let frame_buffer = self.device.get_frame_buffer(&self.render_pass, &frame)?;

            let _pass = encoder.begin_render_pass(
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

            Ok(())
        })?;

        self.device
            .execute(asche::QueueType::Graphics, &[&self.command_buffer])?;
        self.device.wait(asche::WaitForQueueType::Graphics)?;

        self.command_buffer.reset()?;

        self.device.queue_frame(frame)?;

        Ok(())
    }
}
