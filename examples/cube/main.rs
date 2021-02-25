use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::f32::{Vec2, Vec3, Vec4};
use raw_window_handle::HasRawWindowHandle;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: Vec4,
    tex_coord: Vec2,
}

unsafe impl Pod for Vertex {}

unsafe impl Zeroable for Vertex {}

fn main() -> Result<(), asche::AscheError> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .with_title("asche - cube example")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    // Log level is based on RUST_LOG env var.
    #[cfg(feature = "tracing")]
    {
        let filter = tracing_subscriber::EnvFilter::from_default_env();
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    let instance = asche::Instance::new(asche::InstanceConfiguration {
        app_name: "cube example",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &window.raw_window_handle(),
        extensions: vec![],
    })?;

    let (device, (_, graphics_queue, transfer_queue)) =
        instance.request_device(asche::DeviceConfiguration {
            ..Default::default()
        })?;

    let mut app = Application::new(device, graphics_queue, transfer_queue, window)?;

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
    transfer_queue: asche::TransferQueue,
    graphics_command_pool: asche::GraphicsCommandPool,
    window: winit::window::Window,
    extent: vk::Extent2D,
    graphics_pipeline: asche::GraphicsPipeline,
    render_pass: asche::RenderPass,
    frame_counter: u64,
    transfer_counter: u64,
    vertex_buffer: Vec<asche::Buffer>,
    index_buffer: Vec<asche::Buffer>,
    vp_matrix: glam::f32::Mat4,
}

impl Application {
    fn new(
        mut device: asche::Device,
        mut graphics_queue: asche::GraphicsQueue,
        transfer_queue: asche::TransferQueue,
        window: winit::window::Window,
    ) -> Result<Self, asche::AscheError> {
        let extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        // Shader
        let vert_module = device.create_shader_module(
            "vertex module",
            include_bytes!("../../gen/shader/cube.vert.spv"),
        )?;
        let frag_module = device.create_shader_module(
            "fragment module",
            include_bytes!("../../gen/shader/cube.frag.spv"),
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
        let vertex_binding_descriptions = [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()];

        let vertex_attribute_descriptions = [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .offset(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .offset(12)
                .format(vk::Format::R32G32_SFLOAT)
                .build(),
        ];

        let shader_stages = vec![vertexshader_stage.build(), fragmentshader_stage.build()];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);
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
        let pipeline = device.create_graphics_pipeline("cube pipeline", pipeline_info)?;

        let graphics_command_pool = graphics_queue.create_command_pool()?;

        let p_matrix = glam::f32::Mat4::perspective_infinite_reverse_rh(
            (70.0f32).to_radians(),
            extent.width as f32 / extent.height as f32,
            0.1,
        );
        let v_matrix = glam::f32::Mat4::look_at_rh(
            Vec3::new(0.0, -2.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );

        let vp_matrix = p_matrix * v_matrix;

        let mut app = Self {
            device,
            graphics_queue,
            transfer_queue,
            graphics_command_pool,
            window,
            extent,
            graphics_pipeline: pipeline,
            render_pass,
            frame_counter: 0,
            transfer_counter: 0,
            vertex_buffer: vec![],
            index_buffer: vec![],
            vp_matrix,
        };

        let (vertex_data, index_data) = create_cube_data();

        let vertex_buffer = app.create_buffer(
            &bytemuck::cast_slice(&vertex_data),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        let index_buffer = app.create_buffer(
            &bytemuck::cast_slice(&index_data),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        app.vertex_buffer.push(vertex_buffer);
        app.index_buffer.push(index_buffer);

        Ok(app)
    }

    fn create_buffer(
        &mut self,
        buffer_data: &[u8],
        buffer_type: vk::BufferUsageFlags,
    ) -> Result<asche::Buffer, asche::AscheError> {
        let mut stagging_buffer = self.device.create_buffer(
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_alloc::MemoryLocation::CpuToGpu,
            vk::SharingMode::CONCURRENT,
            vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            buffer_data.len() as u64,
            None,
        )?;

        let stagging_slice = stagging_buffer
            .allocation
            .mapped_slice_mut()
            .expect("staging buffer allocation was not mapped");
        stagging_slice[..buffer_data.len()].clone_from_slice(bytemuck::cast_slice(&buffer_data));

        let dst_buffer = self.device.create_buffer(
            buffer_type | vk::BufferUsageFlags::TRANSFER_DST,
            vk_alloc::MemoryLocation::GpuOnly,
            vk::SharingMode::CONCURRENT,
            vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            stagging_buffer.allocation.size,
            None,
        )?;

        let mut transfer_pool = self.transfer_queue.create_command_pool()?;
        let transfer_buffer = transfer_pool
            .create_command_buffer(self.transfer_counter, self.transfer_counter + 1)?;

        transfer_buffer.record(|encoder| {
            encoder.cmd_copy_buffer(
                &stagging_buffer,
                &dst_buffer,
                0,
                0,
                buffer_data.len() as u64,
            );
            Ok(())
        })?;
        self.transfer_counter += 1;

        self.transfer_queue.execute(&transfer_buffer)?;
        self.transfer_queue
            .wait_for_timeline_value(self.transfer_counter)?;

        Ok(dst_buffer)
    }

    fn render(&mut self) -> Result<(), asche::AscheError> {
        let frame_offset = self.frame_counter * Timeline::RenderEnd as u64;
        let frame = self.device.get_next_frame()?;

        let graphics_buffer = self.graphics_command_pool.create_command_buffer(
            Timeline::RenderStart.with_offset(frame_offset),
            Timeline::RenderEnd.with_offset(frame_offset),
        )?;

        let frame_buffer = self.device.get_frame_buffer(&self.render_pass, &frame)?;

        graphics_buffer.record(|encoder| {
            encoder.set_viewport_and_scissor(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            });

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

            pass.cmd_bind_index_buffer(self.index_buffer[0].raw, 0, vk::IndexType::UINT32);
            pass.cmd_bind_vertex_buffer(0, &[self.vertex_buffer[0].raw], &[0]);

            pass.cmd_draw_indexed(36, 1, 0, 0, 0);

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

fn create_cube_data() -> (Vec<Vertex>, Vec<u32>) {
    let vertex_data = [
        Vertex {
            position: Vec4::new(-1.0, -1.0, 1.0, 1.0),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        Vertex {
            position: Vec4::new(1.0, -1.0, 1.0, 1.0),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        Vertex {
            position: Vec4::new(1.0, 1.0, 1.0, 1.0),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        Vertex {
            position: Vec4::new(-1.0, 1.0, 1.0, 1.0),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        Vertex {
            position: Vec4::new(-1.0, 1.0, -1.0, 1.0),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        Vertex {
            position: Vec4::new(1.0, 1.0, -1.0, 1.0),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        Vertex {
            position: Vec4::new(1.0, -1.0, -1.0, 1.0),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        Vertex {
            position: Vec4::new(-1.0, -1.0, -1.0, 1.0),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        Vertex {
            position: Vec4::new(1.0, -1.0, -1.0, 1.0),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        Vertex {
            position: Vec4::new(1.0, 1.0, -1.0, 1.0),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        Vertex {
            position: Vec4::new(1.0, 1.0, 1.0, 1.0),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        Vertex {
            position: Vec4::new(1.0, -1.0, 1.0, 1.0),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        Vertex {
            position: Vec4::new(-1.0, -1.0, 1.0, 1.0),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        Vertex {
            position: Vec4::new(-1.0, 1.0, 1.0, 1.0),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        Vertex {
            position: Vec4::new(-1.0, 1.0, -1.0, 1.0),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        Vertex {
            position: Vec4::new(-1.0, -1.0, -1.0, 1.0),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        Vertex {
            position: Vec4::new(1.0, 1.0, -1.0, 1.0),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        Vertex {
            position: Vec4::new(-1.0, 1.0, -1.0, 1.0),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        Vertex {
            position: Vec4::new(-1.0, 1.0, 1.0, 1.0),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        Vertex {
            position: Vec4::new(1.0, 1.0, 1.0, 1.0),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        Vertex {
            position: Vec4::new(1.0, -1.0, 1.0, 1.0),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        Vertex {
            position: Vec4::new(-1.0, -1.0, 1.0, 1.0),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        Vertex {
            position: Vec4::new(-1.0, -1.0, -1.0, 1.0),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        Vertex {
            position: Vec4::new(1.0, -1.0, -1.0, 1.0),
            tex_coord: Vec2::new(0.0, 1.0),
        },
    ];

    let index_data: &[u32] = &[
        0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, 16, 17,
        18, 18, 19, 16, 20, 21, 22, 22, 23, 20,
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}
