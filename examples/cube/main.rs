use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::f32::{Mat4, Vec2, Vec3, Vec4};
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
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem
        .window("asche - cube example", 1920, 1080)
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
        app_name: "cube example",
        app_version: ash::vk::make_version(1, 0, 0),
        handle: &window.raw_window_handle(),
        extensions: vec![],
    })?;

    let (device, (_, graphics_queue, transfer_queue)) =
        instance.request_device(asche::DeviceConfiguration {
            ..Default::default()
        })?;

    let mut app = Application::new(device, graphics_queue, transfer_queue, &window)?;

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
    transfer_queue: asche::TransferQueue,
    graphics_command_pool: asche::GraphicsCommandPool,
    extent: vk::Extent2D,
    pipeline: asche::GraphicsPipeline,
    pipeline_layout: asche::PipelineLayout,
    render_pass: asche::RenderPass,
    _depth_image: asche::Image,
    depth_image_view: asche::ImageView,
    vertex_buffer: Vec<asche::Buffer>,
    index_buffer: Vec<asche::Buffer>,
    textures: Vec<Texture>,
    sampler: asche::Sampler,
    vp_matrix: glam::f32::Mat4,
    timeline: asche::TimelineSemaphore,
    timeline_value: u64,
    descriptor_set_layout: asche::DescriptorSetLayout,
    descriptor_pool: asche::DescriptorPool,
}

impl Application {
    fn new(
        mut device: asche::Device,
        mut graphics_queue: asche::GraphicsQueue,
        transfer_queue: asche::TransferQueue,
        window: &sdl2::video::Window,
    ) -> Result<Self, asche::AscheError> {
        let (width, height) = window.size();
        let extent = vk::Extent2D { width, height };

        // Shader
        let vert_module = device.create_shader_module(
            "Vertex Shader Module",
            include_bytes!("../../gen/shader/cube.vert.spv"),
        )?;
        let frag_module = device.create_shader_module(
            "Fragment Shader Module",
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

        // Depth image
        let depth_image = device.create_image(&asche::ImageDescriptor {
            name: "Depth Texture",
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queues: vk::QueueFlags::GRAPHICS,
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::D32_SFLOAT,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            initial_layout: vk::ImageLayout::UNDEFINED,
            flags: None,
        })?;

        let depth_image_view = device.create_image_view(&asche::ImageViewDescriptor {
            name: "Depth Texture View",
            image: &depth_image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::D32_SFLOAT,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            flags: None,
        })?;

        // Sampler
        let sampler = device.create_sampler(&asche::SamplerDescriptor {
            name: "Cube Texture Sampler",
            ..Default::default()
        })?;

        // Renderpass
        let attachments = [
            // Color
            vk::AttachmentDescription::builder()
                .format(vk::Format::B8G8R8A8_SRGB)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
            // Depth
            vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
                .stencil_store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
        ];

        let color_attachment_references = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let depth_attachment_references = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_references)
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

        // Descriptor set layout
        let bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1) // Used fore texture arrays
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build()];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let descriptor_set_layout =
            device.create_descriptor_set_layout("Cube Descriptor Set Layout", layout_info)?;

        // Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize::builder()
            .descriptor_count(1)
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build()];

        let descriptor_pool = device.create_descriptor_pool(&asche::DescriptorPoolDescriptor {
            name: "Cube Descriptor Pool",
            max_sets: 16,
            pool_sizes: &pool_sizes,
            flags: None,
        })?;

        // Pipeline layout
        let push_constants_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(64)
            .build()];

        let layouts = [descriptor_set_layout.raw];
        let pipeline_layout = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&push_constants_ranges)
            .set_layouts(&layouts);
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
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .offset(16)
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
            .cull_mode(vk::CullModeFlags::BACK)
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

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .dynamic_state(&dynamic_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .layout(pipeline_layout.raw)
            .render_pass(render_pass.raw)
            .subpass(0);

        let pipeline = device.create_graphics_pipeline("Cube Pipeline", pipeline_info)?;

        let graphics_command_pool = graphics_queue.create_command_pool()?;

        let p_matrix = perspective_infinite_reverse_rh_yup(
            (70.0f32).to_radians(),
            extent.width as f32 / extent.height as f32,
            0.1,
        );
        let v_matrix = Mat4::look_at_rh(Vec3::new(0.0, 2.0, 3.0), Vec3::zero(), Vec3::unit_y());
        let vp_matrix = p_matrix * v_matrix;

        let timeline_value = 0;
        let timeline = device.create_timeline_semaphore("Render Timeline", timeline_value)?;

        let mut app = Self {
            device,
            graphics_queue,
            transfer_queue,
            graphics_command_pool,
            extent,
            pipeline,
            pipeline_layout,
            render_pass,
            _depth_image: depth_image,
            depth_image_view,
            vertex_buffer: vec![],
            index_buffer: vec![],
            textures: vec![],
            sampler,
            vp_matrix,
            timeline,
            timeline_value,
            descriptor_set_layout,
            descriptor_pool,
        };

        // Upload the model data
        let (vertex_data, index_data) = create_cube_data();
        let index_buffer = app.create_buffer(
            "Index Buffer",
            &bytemuck::cast_slice(&index_data),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        let vertex_buffer = app.create_buffer(
            "Vertex Buffer",
            &bytemuck::cast_slice(&vertex_data),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        app.vertex_buffer.push(vertex_buffer);
        app.index_buffer.push(index_buffer);

        // Upload the model texture
        let texture_data = include_bytes!("fractal.dds");
        let texture = app.create_texture("Cube Texture", texture_data)?;
        app.textures.push(texture);

        Ok(app)
    }

    fn create_texture(
        &mut self,
        name: &str,
        image_data: &[u8],
    ) -> Result<Texture, asche::AscheError> {
        let dds = ddsfile::Dds::read(&mut std::io::Cursor::new(&image_data)).unwrap();

        let mut stagging_buffer = self.device.create_buffer(&asche::BufferDescriptor {
            name: "Staging Buffer",
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_location: vk_alloc::MemoryLocation::CpuToGpu,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            size: dds.data.len() as u64,
            flags: None,
        })?;

        let stagging_slice = stagging_buffer
            .allocation
            .mapped_slice_mut()
            .expect("staging buffer allocation was not mapped");
        stagging_slice[..dds.data.len()].clone_from_slice(bytemuck::cast_slice(&dds.data));

        let extent = vk::Extent3D {
            width: dds.header.width,
            height: dds.header.height,
            depth: 1,
        };

        let image = self.device.create_image(&asche::ImageDescriptor {
            name,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::BC7_SRGB_BLOCK,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            initial_layout: vk::ImageLayout::UNDEFINED,
            flags: None,
        })?;

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let view = self.device.create_image_view(&asche::ImageViewDescriptor {
            name: &format!("{} View", name),
            image: &image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::BC7_SRGB_BLOCK,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range,
            flags: None,
        })?;

        let mut transfer_pool = self.transfer_queue.create_command_pool()?;
        let transfer_buffer = transfer_pool.create_command_buffer(
            &self.timeline,
            self.timeline_value,
            self.timeline_value + 1,
        )?;

        transfer_buffer.record(|encoder| {
            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image.raw)
                .subresource_range(subresource_range)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

            encoder.pipeline_barrier(
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier.build()],
            );

            encoder.copy_buffer_to_image(
                &stagging_buffer,
                &image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: extent,
                },
            );
            Ok(())
        })?;
        self.timeline_value += 1;
        self.transfer_queue.execute(&transfer_buffer)?;

        let graphics_buffer = self.graphics_command_pool.create_command_buffer(
            &self.timeline,
            self.timeline_value,
            self.timeline_value + 1,
        )?;

        graphics_buffer.record(|encoder| {
            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image.raw)
                .subresource_range(subresource_range)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            encoder.pipeline_barrier(
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier.build()],
            );

            Ok(())
        })?;

        self.timeline_value += 1;
        self.graphics_queue.execute(&graphics_buffer)?;

        self.timeline.wait_for_value(self.timeline_value)?;

        Ok(Texture {
            _image: image,
            view,
        })
    }

    fn create_buffer(
        &mut self,
        name: &str,
        buffer_data: &[u8],
        buffer_type: vk::BufferUsageFlags,
    ) -> Result<asche::Buffer, asche::AscheError> {
        let mut stagging_buffer = self.device.create_buffer(&asche::BufferDescriptor {
            name: "Staging Buffer",
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_location: vk_alloc::MemoryLocation::CpuToGpu,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            size: buffer_data.len() as u64,
            flags: None,
        })?;

        let stagging_slice = stagging_buffer
            .allocation
            .mapped_slice_mut()
            .expect("staging buffer allocation was not mapped");
        stagging_slice[..buffer_data.len()].clone_from_slice(bytemuck::cast_slice(&buffer_data));

        let dst_buffer = self.device.create_buffer(&asche::BufferDescriptor {
            name,
            usage: buffer_type | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            size: stagging_buffer.allocation.size,
            flags: None,
        })?;

        let mut transfer_pool = self.transfer_queue.create_command_pool()?;
        let transfer_buffer = transfer_pool.create_command_buffer(
            &self.timeline,
            self.timeline_value,
            self.timeline_value + 1,
        )?;

        transfer_buffer.record(|encoder| {
            encoder.copy_buffer(
                &stagging_buffer,
                &dst_buffer,
                0,
                0,
                buffer_data.len() as u64,
            );
            Ok(())
        })?;
        self.timeline_value += 1;

        self.transfer_queue.execute(&transfer_buffer)?;
        self.timeline.wait_for_value(self.timeline_value)?;

        transfer_pool.reset()?;
        self.graphics_command_pool.reset()?;

        Ok(dst_buffer)
    }

    fn render(&mut self) -> Result<(), asche::AscheError> {
        let frame = self.device.get_next_frame()?;

        let graphics_buffer = self.graphics_command_pool.create_command_buffer(
            &self.timeline,
            Timeline::RenderStart.with_offset(self.timeline_value),
            Timeline::RenderEnd.with_offset(self.timeline_value),
        )?;

        let m_matrix =
            Mat4::from_rotation_y((std::f32::consts::PI) * self.timeline_value as f32 / 500.0);
        let mvp_matrix = self.vp_matrix * m_matrix;

        let set = self
            .descriptor_pool
            .create_descriptor_set("Cube Descriptor Set", &self.descriptor_set_layout)?;

        let texture = &self.textures[0];
        set.update(&asche::UpdateDescriptorSetDescriptor {
            binding: 0,
            update: asche::DescriptorSetUpdate::CombinedImageSampler {
                sampler: &self.sampler,
                image_view: &texture.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        });

        graphics_buffer.record(|encoder| {
            encoder.set_viewport_and_scissor(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            });

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
                Some(&asche::RenderPassDepthAttachmentDescriptor {
                    attachment: self.depth_image_view.raw,
                    clear_value: vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [1.0, 0.0, 1.0, 1.0],
                        },
                    },
                }),
                self.extent,
            )?;

            pass.bind_pipeline(&self.pipeline);
            pass.bind_descriptor_set(&self.pipeline_layout, 0, &set, &[]);

            pass.bind_index_buffer(self.index_buffer[0].raw, 0, vk::IndexType::UINT32);
            pass.bind_vertex_buffer(0, &[self.vertex_buffer[0].raw], &[0]);

            pass.push_constants(
                self.pipeline_layout.raw,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(mvp_matrix.as_ref()),
            );

            pass.draw_indexed(36, 1, 0, 0, 0);

            Ok(())
        })?;

        self.graphics_queue.execute(&graphics_buffer)?;
        self.timeline
            .wait_for_value(Timeline::RenderEnd.with_offset(self.timeline_value))?;

        self.graphics_command_pool.reset()?;
        self.descriptor_pool.free_sets()?;

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

/// Right-handed with the the x-axis pointing right, y-axis pointing up, and z-axis pointing out of the screen for Vulkan NDC.
#[inline]
fn perspective_infinite_reverse_rh_yup(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Mat4 {
    let f = 1.0 / (0.5 * fov_y_radians).tan();
    Mat4::from_cols(
        Vec4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
        Vec4::new(0.0, -f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, z_near, 0.0),
    )
}

struct Texture {
    _image: asche::Image,
    view: asche::ImageView,
}
