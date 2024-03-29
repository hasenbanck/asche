use bytemuck::{cast_slice, Pod, Zeroable};
use erupt::vk;
use glam::{Mat4, Vec3, Vec4};

use asche::{CommandBufferSemaphore, CommonCommands, QueueConfiguration, Queues};

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 4],
    tex_coord: [f32; 2],
}

unsafe impl Pod for Vertex {}

unsafe impl Zeroable for Vertex {}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum Lifetime {
    Static,
}

impl asche::Lifetime for Lifetime {}

fn main() -> Result<(), asche::AscheError> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("asche - cube example")
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
            app_name: "cube example",
            app_version: asche::Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
            engine_name: "engine example",
            engine_version: asche::Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
            extensions: vec![],
        },
    )?;

    let (device, swapchain, queues) = unsafe {
        instance.request_device(asche::DeviceConfiguration {
            queue_configuration: QueueConfiguration {
                compute_queues: vec![],
                graphics_queues: vec![1.0],
                transfer_queues: vec![1.0],
            },
            ..Default::default()
        })
    }?;

    let Queues {
        compute_queues: _compute_queues,
        mut graphics_queues,
        mut transfer_queues,
    } = queues;

    let mut app = Application::new(
        device,
        swapchain,
        graphics_queues.pop().unwrap(),
        transfer_queues.pop().unwrap(),
        &window,
    )?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        match event {
            winit::event::Event::WindowEvent {
                event:
                    winit::event::WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                state: winit::event::ElementState::Pressed,
                                virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => *control_flow = winit::event_loop::ControlFlow::Exit,
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => *control_flow = winit::event_loop::ControlFlow::Exit,
            winit::event::Event::MainEventsCleared => {
                unsafe { app.render().unwrap() };
            }
            _ => (),
        }
    });
}

struct Application {
    frame_counter: u64,
    extent: vk::Extent2D,
    vp_matrix: Mat4,
    render_fence: asche::Fence,
    presentation_semaphore: asche::BinarySemaphore,
    render_semaphore: asche::BinarySemaphore,
    transfer_timeline: asche::TimelineSemaphore,
    transfer_timeline_value: u64,
    descriptor_set_layout: asche::DescriptorSetLayout,
    descriptor_pool: asche::DescriptorPool,
    graphics_command_pool: asche::GraphicsCommandPool,
    pipeline: asche::GraphicsPipeline,
    pipeline_layout: asche::PipelineLayout,
    render_pass: asche::RenderPass,
    vertex_buffer: Vec<asche::Buffer<Lifetime>>,
    index_buffer: Vec<asche::Buffer<Lifetime>>,
    sampler: asche::Sampler,
    depth_texture: Texture,
    textures: Vec<Texture>,
    graphics_queue: asche::GraphicsQueue,
    transfer_queue: asche::TransferQueue,
    swapchain: asche::Swapchain,
    device: asche::Device<Lifetime>,
}

impl Drop for Application {
    fn drop(&mut self) {
        unsafe {
            self.device
                .wait_idle()
                .expect("couldn't wait for device to become idle while dropping");
        }
    }
}

impl Application {
    fn new(
        device: asche::Device<Lifetime>,
        swapchain: asche::Swapchain,
        mut graphics_queue: asche::GraphicsQueue,
        transfer_queue: asche::TransferQueue,
        window: &winit::window::Window,
    ) -> Result<Self, asche::AscheError> {
        let extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        // Shader
        let vert_module = unsafe {
            device.create_shader_module(
                "Vertex Shader Module",
                include_bytes!("shader/cube.vert.spv"),
            )
        }?;
        let frag_module = unsafe {
            device.create_shader_module(
                "Fragment Shader Module",
                include_bytes!("shader/cube.frag.spv"),
            )
        }?;

        let mainfunctionname = std::ffi::CString::new("main").unwrap();
        let vertexshader_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(vert_module.raw())
            .name(&mainfunctionname);
        let fragmentshader_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(frag_module.raw())
            .name(&mainfunctionname);

        // Depth image
        let depth_image = unsafe {
            device.create_image(&asche::ImageDescriptor::<_> {
                name: "Depth Texture",
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                memory_location: vk_alloc::MemoryLocation::GpuOnly,
                lifetime: Lifetime::Static,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queues: vk::QueueFlags::GRAPHICS,
                image_type: vk::ImageType::_2D,
                format: vk::Format::D32_SFLOAT,
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlagBits::_1,
                tiling: vk::ImageTiling::OPTIMAL,
                initial_layout: vk::ImageLayout::UNDEFINED,
                flags: None,
            })
        }?;

        let depth_image_view = unsafe {
            device.create_image_view(&asche::ImageViewDescriptor {
                name: "Depth Texture View",
                image: &depth_image,
                view_type: vk::ImageViewType::_2D,
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
            })
        }?;

        let depth_texture = Texture {
            view: depth_image_view,
            _image: depth_image,
        };

        // Sampler
        let sampler = unsafe {
            device.create_sampler(&asche::SamplerDescriptor {
                name: "Cube Texture Sampler",
                ..Default::default()
            })
        }?;

        // Renderpass
        let attachments = [
            // Color
            vk::AttachmentDescription2Builder::new()
                .format(vk::Format::B8G8R8A8_SRGB)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .samples(vk::SampleCountFlagBits::_1),
            // Depth
            vk::AttachmentDescription2Builder::new()
                .format(vk::Format::D32_SFLOAT)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .samples(vk::SampleCountFlagBits::_1),
        ];

        let color_attachment_references = [vk::AttachmentReference2Builder::new()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let depth_attachment_references = vk::AttachmentReference2Builder::new()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpasses = [vk::SubpassDescription2Builder::new()
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_references)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];

        let renderpass_info = vk::RenderPassCreateInfo2Builder::new()
            .attachments(&attachments)
            .subpasses(&subpasses);

        let render_pass =
            unsafe { device.create_render_pass("Graphics Render Pass Simple", renderpass_info) }?;

        // Descriptor set layout
        let bindings = [vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_count(1) // Used fore texture arrays
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout("Cube Descriptor Set Layout", layout_info)
        }?;

        // Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSizeBuilder::new()
            .descriptor_count(1)
            ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)];

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&asche::DescriptorPoolDescriptor {
                name: "Cube Descriptor Pool",
                max_sets: 16,
                pool_sizes: &pool_sizes,
                flags: None,
            })
        }?;

        // Pipeline layout
        let push_constants_ranges = [vk::PushConstantRangeBuilder::new()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(64)];

        let layouts = [descriptor_set_layout.raw()];
        let pipeline_layout = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&push_constants_ranges)
            .set_layouts(&layouts);
        let pipeline_layout =
            unsafe { device.create_pipeline_layout("Pipeline Layout Simple", pipeline_layout) }?;

        // Pipeline
        let vertex_binding_descriptions = [vk::VertexInputBindingDescriptionBuilder::new()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];

        let vertex_attribute_descriptions = [
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(0)
                .offset(0)
                .format(vk::Format::R32G32B32A32_SFLOAT),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(1)
                .offset(16)
                .format(vk::Format::R32G32_SFLOAT),
        ];

        let shader_stages = vec![vertexshader_stage, fragmentshader_stage];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfoBuilder::new()
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);
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
            .cull_mode(vk::CullModeFlags::BACK)
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

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        let pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .dynamic_state(&dynamic_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .layout(pipeline_layout.raw())
            .render_pass(render_pass.raw())
            .subpass(0);

        let pipeline = unsafe { device.create_graphics_pipeline("Cube Pipeline", pipeline_info) }?;

        let graphics_command_pool = unsafe { graphics_queue.create_command_pool() }?;

        let p_matrix = perspective_infinite_reverse_rh_yup(
            (70.0f32).to_radians(),
            extent.width as f32 / extent.height as f32,
            0.1,
        );
        let v_matrix = Mat4::look_at_rh(Vec3::new(0.0, 2.0, 3.0), Vec3::ZERO, Vec3::Y);
        let vp_matrix = p_matrix * v_matrix;

        let render_fence = unsafe { device.create_fence("Render Fence") }?;
        let presentation_semaphore =
            unsafe { device.create_binary_semaphore("Presentation Semaphore") }?;
        let render_semaphore = unsafe { device.create_binary_semaphore("Render Semaphore") }?;

        let transfer_timeline_value = 0;
        let transfer_timeline = unsafe {
            device.create_timeline_semaphore("Transfer Timeline", transfer_timeline_value)
        }?;

        let mut app = Self {
            frame_counter: 0,
            device,
            swapchain,
            graphics_queue,
            transfer_queue,
            graphics_command_pool,
            extent,
            pipeline,
            pipeline_layout,
            render_pass,
            vertex_buffer: vec![],
            index_buffer: vec![],
            textures: vec![],
            depth_texture,
            sampler,
            vp_matrix,
            render_fence,
            presentation_semaphore,
            render_semaphore,
            transfer_timeline,
            transfer_timeline_value,
            descriptor_set_layout,
            descriptor_pool,
        };

        // Upload the model data
        let (vertex_data, index_data) = create_cube_data();
        let index_buffer = unsafe {
            app.create_buffer(
                "Index Buffer",
                bytemuck::cast_slice(&index_data),
                vk::BufferUsageFlags::INDEX_BUFFER,
            )
        }?;

        let vertex_buffer = unsafe {
            app.create_buffer(
                "Vertex Buffer",
                bytemuck::cast_slice(&vertex_data),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )
        }?;

        app.vertex_buffer.push(vertex_buffer);
        app.index_buffer.push(index_buffer);

        // Upload the model texture
        let texture_data = include_bytes!("fractal.dds");
        let texture = unsafe { app.create_texture("Cube Texture", texture_data) }?;
        app.textures.push(texture);

        Ok(app)
    }

    unsafe fn create_texture(
        &mut self,
        name: &str,
        image_data: &[u8],
    ) -> Result<Texture, asche::AscheError> {
        let dds = ddsfile::Dds::read(&mut std::io::Cursor::new(&image_data)).unwrap();

        let mut stagging_buffer = self.device.create_buffer(&asche::BufferDescriptor::<_> {
            name: "Staging Buffer",
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_location: vk_alloc::MemoryLocation::CpuToGpu,
            lifetime: Lifetime::Static,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            size: dds.data.len() as u64,
            flags: None,
        })?;

        let stagging_slice = stagging_buffer
            .mapped_slice_mut()?
            .expect("staging buffer allocation was not mapped");
        stagging_slice[..dds.data.len()].clone_from_slice(bytemuck::cast_slice(&dds.data));
        stagging_buffer.flush()?;

        let extent = vk::Extent3D {
            width: dds.header.width,
            height: dds.header.height,
            depth: 1,
        };

        let image = self.device.create_image(&asche::ImageDescriptor::<_> {
            name,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            lifetime: Lifetime::Static,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            image_type: vk::ImageType::_2D,
            format: vk::Format::BC7_SRGB_BLOCK,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlagBits::_1,
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
            view_type: vk::ImageViewType::_2D,
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
            &[CommandBufferSemaphore::Timeline {
                semaphore: self.transfer_timeline.handle(),
                stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                value: self.transfer_timeline_value,
            }],
            &[CommandBufferSemaphore::Timeline {
                semaphore: self.transfer_timeline.handle(),
                stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                value: self.transfer_timeline_value + 1,
            }],
        )?;

        {
            let encoder = transfer_buffer.record()?;
            let barrier = [vk::ImageMemoryBarrier2KHRBuilder::new()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image.raw())
                .subresource_range(subresource_range)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::TRANSFER_KHR)
                .dst_access_mask(vk::AccessFlags2KHR::TRANSFER_WRITE_KHR)];

            let dependency_info = vk::DependencyInfoKHRBuilder::new()
                .memory_barriers(&[])
                .image_memory_barriers(&barrier)
                .buffer_memory_barriers(&[]);

            encoder.pipeline_barrier2(&dependency_info);

            encoder.copy_buffer_to_image(
                stagging_buffer.raw(),
                image.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::BufferImageCopyBuilder::new()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(extent),
            );

            let barrier = [vk::ImageMemoryBarrier2KHRBuilder::new()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image.raw())
                .subresource_range(subresource_range)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_stage_mask(vk::PipelineStageFlags2KHR::TRANSFER_KHR)
                .src_access_mask(vk::AccessFlags2KHR::TRANSFER_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_TRANSFER_KHR)
                .dst_access_mask(vk::AccessFlags2KHR::NONE_KHR)];

            let dependency_info = vk::DependencyInfoKHRBuilder::new()
                .memory_barriers(&[])
                .image_memory_barriers(&barrier)
                .buffer_memory_barriers(&[]);

            encoder.pipeline_barrier2(&dependency_info);
        }

        self.transfer_timeline_value += 1;
        self.transfer_queue.submit(&transfer_buffer, None)?;
        self.transfer_timeline
            .wait_for_value(self.transfer_timeline_value)?;

        Ok(Texture {
            view,
            _image: image,
        })
    }

    unsafe fn create_buffer(
        &mut self,
        name: &str,
        buffer_data: &[u8],
        buffer_type: vk::BufferUsageFlags,
    ) -> Result<asche::Buffer<Lifetime>, asche::AscheError> {
        let mut stagging_buffer = self.device.create_buffer(&asche::BufferDescriptor::<_> {
            name: "Staging Buffer",
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_location: vk_alloc::MemoryLocation::CpuToGpu,
            lifetime: Lifetime::Static,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            size: buffer_data.len() as u64,
            flags: None,
        })?;

        let stagging_slice = stagging_buffer
            .mapped_slice_mut()?
            .expect("staging buffer allocation was not mapped");
        stagging_slice[..buffer_data.len()].clone_from_slice(bytemuck::cast_slice(buffer_data));

        let dst_buffer = self.device.create_buffer(&asche::BufferDescriptor::<_> {
            name,
            usage: buffer_type | vk::BufferUsageFlags::TRANSFER_DST,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            lifetime: Lifetime::Static,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS,
            size: buffer_data.len() as u64,
            flags: None,
        })?;

        let mut transfer_pool = self.transfer_queue.create_command_pool()?;
        let transfer_buffer = transfer_pool.create_command_buffer(
            &[CommandBufferSemaphore::Timeline {
                semaphore: self.transfer_timeline.handle(),
                stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                value: self.transfer_timeline_value,
            }],
            &[CommandBufferSemaphore::Timeline {
                semaphore: self.transfer_timeline.handle(),
                stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                value: self.transfer_timeline_value + 1,
            }],
        )?;

        {
            let encoder = transfer_buffer.record()?;
            encoder.copy_buffer(
                stagging_buffer.raw(),
                dst_buffer.raw(),
                0,
                0,
                buffer_data.len() as u64,
            );
        }

        self.transfer_timeline_value += 1;
        self.transfer_queue.submit(&transfer_buffer, None)?;
        self.transfer_timeline
            .wait_for_value(self.transfer_timeline_value)?;

        transfer_pool.reset()?;
        self.graphics_command_pool.reset()?;

        Ok(dst_buffer)
    }

    unsafe fn render(&mut self) -> Result<(), asche::AscheError> {
        let frame = self.swapchain.next_frame(&self.presentation_semaphore)?;

        let graphics_buffer = self.graphics_command_pool.create_command_buffer(
            &[],
            &[CommandBufferSemaphore::Binary {
                semaphore: self.render_semaphore.handle(),
                stage: vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT_KHR,
            }],
        )?;

        let m_matrix =
            Mat4::from_rotation_y((std::f32::consts::PI) * self.frame_counter as f32 / 500.0);
        let mvp_matrix = self.vp_matrix * m_matrix;

        let set = self.descriptor_pool.create_descriptor_set(
            "Cube Descriptor Set",
            &self.descriptor_set_layout,
            None,
        )?;

        let image_info = [vk::DescriptorImageInfoBuilder::new()
            .sampler(self.sampler.raw())
            .image_view(self.textures[0].view.raw())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(set.raw())
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info);
        self.device.update_descriptor_sets(&[write], &[]);

        {
            let encoder = graphics_buffer.record()?;
            encoder.set_viewport_and_scissor(
                vk::Rect2DBuilder::new()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(self.extent),
            );

            let pass = encoder.begin_render_pass(
                &mut self.swapchain,
                &self.render_pass,
                &[asche::RenderPassColorAttachmentDescriptor {
                    attachment: frame.view(),
                    clear_value: Some(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [1.0, 0.0, 1.0, 1.0],
                        },
                    }),
                }],
                Some(asche::RenderPassDepthAttachmentDescriptor {
                    attachment: self.depth_texture.view.raw(),
                    clear_value: Some(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [1.0, 0.0, 1.0, 1.0],
                        },
                    }),
                }),
                self.extent,
            )?;

            pass.bind_pipeline(&self.pipeline);
            pass.bind_descriptor_sets(self.pipeline_layout.raw(), 0, &[set.raw()], &[]);

            pass.bind_index_buffer(self.index_buffer[0].raw(), 0, vk::IndexType::UINT32);
            pass.bind_vertex_buffers(0, &[self.vertex_buffer[0].raw()], &[0]);

            pass.push_constants(
                self.pipeline_layout.raw(),
                vk::ShaderStageFlags::VERTEX,
                0,
                cast_slice(mvp_matrix.as_ref()),
            )?;

            pass.draw_indexed(36, 1, 0, 0, 0);
        }

        self.graphics_queue
            .submit(&graphics_buffer, Some(&self.render_fence))?;

        self.render_fence.wait()?;
        self.render_fence.reset()?;

        self.graphics_command_pool.reset()?;
        self.descriptor_pool.free_sets()?;

        self.swapchain.queue_frame(
            &self.graphics_queue,
            frame,
            &[&self.presentation_semaphore, &self.render_semaphore],
        )?;

        self.frame_counter += 1;

        Ok(())
    }
}

fn create_cube_data() -> (Vec<Vertex>, Vec<u32>) {
    let vertex_data = [
        Vertex {
            position: [-1.0, -1.0, 1.0, 1.0],
            tex_coord: [0.0, 0.0],
        },
        Vertex {
            position: [1.0, -1.0, 1.0, 1.0],
            tex_coord: [1.0, 0.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0, 1.0],
            tex_coord: [1.0, 1.0],
        },
        Vertex {
            position: [-1.0, 1.0, 1.0, 1.0],
            tex_coord: [0.0, 1.0],
        },
        Vertex {
            position: [-1.0, 1.0, -1.0, 1.0],
            tex_coord: [1.0, 0.0],
        },
        Vertex {
            position: [1.0, 1.0, -1.0, 1.0],
            tex_coord: [0.0, 0.0],
        },
        Vertex {
            position: [1.0, -1.0, -1.0, 1.0],
            tex_coord: [0.0, 1.0],
        },
        Vertex {
            position: [-1.0, -1.0, -1.0, 1.0],
            tex_coord: [1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0, -1.0, 1.0],
            tex_coord: [0.0, 0.0],
        },
        Vertex {
            position: [1.0, 1.0, -1.0, 1.0],
            tex_coord: [1.0, 0.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0, 1.0],
            tex_coord: [1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0, 1.0, 1.0],
            tex_coord: [0.0, 1.0],
        },
        Vertex {
            position: [-1.0, -1.0, 1.0, 1.0],
            tex_coord: [1.0, 0.0],
        },
        Vertex {
            position: [-1.0, 1.0, 1.0, 1.0],
            tex_coord: [0.0, 0.0],
        },
        Vertex {
            position: [-1.0, 1.0, -1.0, 1.0],
            tex_coord: [0.0, 1.0],
        },
        Vertex {
            position: [-1.0, -1.0, -1.0, 1.0],
            tex_coord: [1.0, 1.0],
        },
        Vertex {
            position: [1.0, 1.0, -1.0, 1.0],
            tex_coord: [1.0, 0.0],
        },
        Vertex {
            position: [-1.0, 1.0, -1.0, 1.0],
            tex_coord: [0.0, 0.0],
        },
        Vertex {
            position: [-1.0, 1.0, 1.0, 1.0],
            tex_coord: [0.0, 1.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0, 1.0],
            tex_coord: [1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0, 1.0, 1.0],
            tex_coord: [0.0, 0.0],
        },
        Vertex {
            position: [-1.0, -1.0, 1.0, 1.0],
            tex_coord: [1.0, 0.0],
        },
        Vertex {
            position: [-1.0, -1.0, -1.0, 1.0],
            tex_coord: [1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0, -1.0, 1.0],
            tex_coord: [0.0, 1.0],
        },
    ];

    let index_data: &[u32] = &[
        0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, 16, 17,
        18, 18, 19, 16, 20, 21, 22, 22, 23, 20,
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

struct Texture {
    view: asche::ImageView,
    _image: asche::Image<Lifetime>,
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
