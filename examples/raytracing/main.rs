use std::time::Duration;

use bytemuck::cast_slice;
use erupt::{vk, ExtendableFrom};

use asche::BufferDescriptor;

use crate::gltf::{Material, Mesh, Vertex};

mod gltf;

type Result<T> = std::result::Result<T, asche::AscheError>;

const SURFACE_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const OFFSCREEN_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

fn main() -> Result<()> {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem
        .window("asche - raytracing example", 1920, 1080)
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
            app_name: "raytracing example",
            app_version: erupt::vk::make_version(1, 0, 0),
            extensions: vec![],
        },
    )?;

    let (device, (compute_queue, graphics_queue, transfer_queue)) =
        instance.request_device(asche::DeviceConfiguration {
            extensions: vec![
                vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                vk::KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
                vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            ],
            features_v1_2: Some(
                vk::PhysicalDeviceVulkan12FeaturesBuilder::new()
                    .timeline_semaphore(true)
                    .buffer_device_address(true)
                    .scalar_block_layout(true)
                    .descriptor_indexing(true)
                    .uniform_buffer_standard_layout(true),
            ),
            features_raytracing: Some(
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHRBuilder::new()
                    .ray_tracing_pipeline(true),
            ),
            features_acceleration_structure: Some(
                vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder::new()
                    .acceleration_structure(true),
            ),
            ..Default::default()
        })?;

    let (width, height) = window.size();

    let mut app = RayTracingApplication::new(
        device,
        compute_queue,
        graphics_queue,
        transfer_queue,
        width,
        height,
    )
    .unwrap();
    let (materials, meshes) = gltf::load_models(include_bytes!("model.glb"));
    app.create_model(&materials, &meshes)?;

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
    }

    Ok(())
}

struct RayTracingApplication {
    models: Vec<Model>,
    timeline: asche::TimelineSemaphore,
    timeline_value: u64,
    offscreen_attachment: Texture,
    postprocess_renderpass: asche::RenderPass,
    postprocess_pipeline_layout: asche::PipelineLayout,
    postprocess_pipeline: asche::GraphicsPipeline,
    postprocess_descriptor_pool: asche::DescriptorPool,
    postprocess_descriptor_set: asche::DescriptorSet,
    raytracing_renderpass: asche::RenderPass,
    raytracing_pipeline_layout: asche::PipelineLayout,
    raytracing_pipeline: asche::RayTracingPipeline,
    raytracing_descriptor_pool: asche::DescriptorPool,
    sampler: asche::Sampler,
    raytrace_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    transfer_pool: asche::TransferCommandPool,
    graphics_pool: asche::GraphicsCommandPool,
    compute_pool: asche::ComputeCommandPool,
    compute_queue: asche::ComputeQueue,
    graphics_queue: asche::GraphicsQueue,
    transfer_queue: asche::TransferQueue,
    device: asche::Device,
}

impl RayTracingApplication {
    fn new(
        mut device: asche::Device,
        mut compute_queue: asche::ComputeQueue,
        mut graphics_queue: asche::GraphicsQueue,
        mut transfer_queue: asche::TransferQueue,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let extent = vk::Extent2D { width, height };

        // Query for RT capabilities
        let mut raytrace_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHRBuilder::new();
        let properties =
            vk::PhysicalDeviceProperties2Builder::new().extend_from(&mut raytrace_properties);
        device.physical_device_properties(properties);

        // Offscreen Attachment
        let offscreen_attachment = Texture::create_offscreen_attachment(&device, extent)?;

        // Sampler
        let sampler = device.create_sampler(&asche::SamplerDescriptor {
            name: "Offscreen Texture Sampler",
            ..Default::default()
        })?;

        let (
            raytracing_renderpass,
            raytracing_descriptor_pool,
            raytracing_pipeline_layout,
            raytracing_pipeline,
        ) = RayTracingApplication::create_rt_renderpass(&mut device, &mut raytrace_properties)?;

        let (
            postprocess_renderpass,
            postprocess_descriptor_pool,
            postprocess_descriptor_set,
            postprocess_pipeline_layout,
            postprocess_pipeline,
        ) = RayTracingApplication::create_postprocess_renderpass(
            &mut device,
            &offscreen_attachment,
            &sampler,
        )?;

        let timeline_value = 0;
        let timeline = device.create_timeline_semaphore("Render Timeline", timeline_value)?;

        let compute_pool = compute_queue.create_command_pool()?;
        let graphics_pool = graphics_queue.create_command_pool()?;
        let transfer_pool = transfer_queue.create_command_pool()?;

        Ok(Self {
            models: vec![],
            timeline,
            timeline_value,
            offscreen_attachment,
            postprocess_renderpass,
            postprocess_pipeline,
            postprocess_pipeline_layout,
            postprocess_descriptor_pool,
            postprocess_descriptor_set,
            raytracing_renderpass,
            raytracing_pipeline,
            raytracing_pipeline_layout,
            raytracing_descriptor_pool,
            raytrace_properties: raytrace_properties.build(),
            transfer_pool,
            graphics_pool,
            sampler,
            compute_queue,
            graphics_queue,
            transfer_queue,
            device,
            compute_pool,
        })
    }

    fn create_rt_renderpass(
        device: &mut asche::Device,
        raytrace_properties: &mut vk::PhysicalDeviceRayTracingPipelinePropertiesKHRBuilder,
    ) -> Result<(
        asche::RenderPass,
        asche::DescriptorPool,
        asche::PipelineLayout,
        asche::RayTracingPipeline,
    )> {
        // RT shader
        let mainfunctionname = std::ffi::CString::new("main").unwrap();
        let raygen_module = device.create_shader_module(
            "Raytrace Raygen Shader Module",
            include_bytes!("shader/raytrace.rgen.spv"),
        )?;
        let close_hit_module = device.create_shader_module(
            "Raytrace Close Hit Shader Module",
            include_bytes!("shader/raytrace.rchit.spv"),
        )?;
        let miss_module = device.create_shader_module(
            "Raytrace Miss Shader Module",
            include_bytes!("shader/raytrace.rmiss.spv"),
        )?;

        let raygen_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::RAYGEN_KHR)
            .module(raygen_module.raw)
            .name(&mainfunctionname);
        let close_hit_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::CLOSEST_HIT_KHR)
            .module(close_hit_module.raw)
            .name(&mainfunctionname);
        let miss_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::MISS_KHR)
            .module(miss_module.raw)
            .name(&mainfunctionname);

        // RT renderpass
        let attachments = [
            // Offscreen Attachment
            vk::AttachmentDescription2Builder::new()
                .format(OFFSCREEN_FORMAT)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .samples(vk::SampleCountFlagBits::_1),
        ];

        let color_attachments = [vk::AttachmentReference2Builder::new()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let subpasses = [vk::SubpassDescription2Builder::new()
            .color_attachments(&color_attachments)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];

        let renderpass_info = vk::RenderPassCreateInfo2Builder::new()
            .attachments(&attachments)
            .subpasses(&subpasses);

        let raytracing_renderpass = device.create_render_pass("RT Render Pass", renderpass_info)?;

        // RT descriptor set layout
        // TODO
        let bindings = [];
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);
        let rt_descriptor_set_layout =
            device.create_descriptor_set_layout("RT Descriptor Set Layout", layout_info)?;

        // RT descriptor pool + set
        // TODO
        let pool_sizes = [vk::DescriptorPoolSizeBuilder::new()
            .descriptor_count(1)
            ._type(vk::DescriptorType::UNIFORM_BUFFER)];
        let raytracing_descriptor_pool =
            device.create_descriptor_pool(&asche::DescriptorPoolDescriptor {
                name: "RT Descriptor Pool",
                max_sets: 2,
                pool_sizes: &pool_sizes,
                flags: None,
            })?;

        // RT pipeline layout
        let push_constants_ranges = [vk::PushConstantRangeBuilder::new()
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .offset(0)
            .size(36)];

        let layouts = [rt_descriptor_set_layout.raw];
        let layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&push_constants_ranges)
            .set_layouts(&layouts);
        let raytracing_pipeline_layout =
            device.create_pipeline_layout("RT Pipeline Layout", layout_info)?;

        // RT Pipeline
        let shader_stages = vec![raygen_stage, miss_stage, close_hit_stage];

        // TODO set a sensible limit
        let max_bounce = raytrace_properties.max_ray_recursion_depth.min(2);

        let groups = [
            vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
                ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
                ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
                ._type(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP_KHR)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR),
        ];

        let rt_pipeline_info = vk::RayTracingPipelineCreateInfoKHRBuilder::new()
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(max_bounce)
            .stages(&shader_stages)
            .layout(raytracing_pipeline_layout.raw);

        let raytracing_pipeline =
            device.create_raytracing_pipeline("RT Pipeline", None, rt_pipeline_info)?;

        Ok((
            raytracing_renderpass,
            raytracing_descriptor_pool,
            raytracing_pipeline_layout,
            raytracing_pipeline,
        ))
    }

    fn create_postprocess_renderpass(
        device: &mut asche::Device,
        offscreen_attachment: &Texture,
        sampler: &asche::Sampler,
    ) -> Result<(
        asche::RenderPass,
        asche::DescriptorPool,
        asche::DescriptorSet,
        asche::PipelineLayout,
        asche::GraphicsPipeline,
    )> {
        // Postprocess shader
        let mainfunctionname = std::ffi::CString::new("main").unwrap();
        let frag_module = device.create_shader_module(
            "Postprocess Fragment Shader Module",
            include_bytes!("shader/postprocess.frag.spv"),
        )?;
        let vert_module = device.create_shader_module(
            "Postprocess Vertex Shader Module",
            include_bytes!("shader/postprocess.vert.spv"),
        )?;

        let frag_module_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(frag_module.raw)
            .name(&mainfunctionname);
        let vert_module_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(vert_module.raw)
            .name(&mainfunctionname);

        // Postprocess renderpass
        let attachments = [
            // Offscreen Attachment
            vk::AttachmentDescription2Builder::new()
                .format(OFFSCREEN_FORMAT)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .samples(vk::SampleCountFlagBits::_1),
            // Output Attachment
            vk::AttachmentDescription2Builder::new()
                .format(SURFACE_FORMAT)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .samples(vk::SampleCountFlagBits::_1),
        ];

        let input_attachments = [vk::AttachmentReference2Builder::new()
            .attachment(0)
            .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .aspect_mask(vk::ImageAspectFlags::COLOR)];

        let color_attachments = [vk::AttachmentReference2Builder::new()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let subpasses = [vk::SubpassDescription2Builder::new()
            .input_attachments(&input_attachments)
            .color_attachments(&color_attachments)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];

        let dependencies = [vk::SubpassDependency2KHRBuilder::new()
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)];

        let renderpass_info = vk::RenderPassCreateInfo2Builder::new()
            .dependencies(&dependencies)
            .attachments(&attachments)
            .subpasses(&subpasses);

        let postprocess_renderpass =
            device.create_render_pass("Postprocess Render Pass", renderpass_info)?;

        // Postprocess descriptor set layout
        let bindings = [vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);
        let postprocess_descriptor_set_layout = device
            .create_descriptor_set_layout("Postprocess Descriptor Set Layout", layout_info)?;

        // Postprocess descriptor pool + set
        let pool_sizes = [vk::DescriptorPoolSizeBuilder::new()
            .descriptor_count(1)
            ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)];
        let postprocess_descriptor_pool =
            device.create_descriptor_pool(&asche::DescriptorPoolDescriptor {
                name: "Postprocess Descriptor Pool",
                max_sets: 2,
                pool_sizes: &pool_sizes,
                flags: None,
            })?;

        let postprocess_descriptor_set = postprocess_descriptor_pool.create_descriptor_set(
            "Postprocess Descriptor Set",
            &postprocess_descriptor_set_layout,
        )?;

        let descriptor_update = asche::UpdateDescriptorSetDescriptor {
            binding: 0,
            update: asche::DescriptorSetUpdate::CombinedImageSampler {
                sampler: &sampler,
                image_view: &offscreen_attachment.view,
                image_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL_KHR,
            },
        };
        postprocess_descriptor_set.update(&descriptor_update);

        // Postprocess pipeline layout
        let layouts = [postprocess_descriptor_set_layout.raw];
        let layout_info = vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&layouts);
        let postprocess_pipeline_layout =
            device.create_pipeline_layout("Postprocess Pipeline Layout", layout_info)?;

        // Postprocess pipeline
        let shader_stages = vec![vert_module_stage, frag_module_stage];
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

        let postprocess_pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .dynamic_state(&dynamic_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(postprocess_pipeline_layout.raw)
            .render_pass(postprocess_renderpass.raw)
            .subpass(0);

        let postprocess_pipeline =
            device.create_graphics_pipeline("Postprocess Pipeline", postprocess_pipeline_info)?;
        Ok((
            postprocess_renderpass,
            postprocess_descriptor_pool,
            postprocess_descriptor_set,
            postprocess_pipeline_layout,
            postprocess_pipeline,
        ))
    }

    /// Upload vertex data and prepares the TLAS and BLAS structures.
    pub fn create_model(&mut self, materials: &[Material], meshes: &[Mesh]) -> Result<()> {
        for (id, mesh) in meshes.iter().enumerate() {
            let vertex_buffer = self
                .create_buffer(
                    &format!("Model {} Vertex Buffer", id),
                    cast_slice(mesh.vertices.as_slice()),
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .unwrap();
            let index_buffer = self
                .create_buffer(
                    &format!("Model {} Index Buffer", id),
                    cast_slice(mesh.indices.as_slice()),
                    vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .unwrap();

            // TODO this could actually also include the material properties!
            let transform_buffer = self
                .create_buffer(
                    &format!("Model {} Transform Buffer", id),
                    cast_slice(&mesh.model_matrix),
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .unwrap();

            self.models.push(Model {
                max_vertex_index: mesh.vertices.len() as u32, // TODO could be one by 1
                triangle_count: (mesh.indices.len() / 3) as u32,
                tlas_index: id as u32,
                blas_index: id as u32,
                transform_buffer,
                vertex_buffer,
                index_buffer,
            })
        }

        self.create_blas()?;

        Ok(())
    }

    fn create_buffer(
        &mut self,
        name: &str,
        buffer_data: &[u8],
        buffer_type: vk::BufferUsageFlags,
    ) -> Result<asche::Buffer> {
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

        let transfer_buffer = self.transfer_pool.create_command_buffer(
            &self.timeline,
            self.timeline_value,
            self.timeline_value + 1,
        )?;

        {
            let encoder = transfer_buffer.record()?;
            encoder.copy_buffer(
                &stagging_buffer,
                &dst_buffer,
                0,
                0,
                buffer_data.len() as u64,
            );
        }

        self.timeline_value += 1;
        self.transfer_queue.submit(&transfer_buffer)?;
        self.timeline.wait_for_value(self.timeline_value)?;

        self.transfer_pool.reset()?;

        Ok(dst_buffer)
    }

    fn create_blas(&mut self) -> Result<()> {
        let mut blas_vec = vec![];

        // Scratchpads needed for the generation of the AS.
        let mut scratch_pads: Vec<asche::Buffer> = vec![];

        // We need to assemble all information that is needed to build the AS so that it outlices the loop were we assemble them.
        let mut geometries: Vec<vk::AccelerationStructureGeometryKHRBuilder> = vec![];
        let mut infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHRBuilder> = vec![];
        let mut ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = vec![];

        // Create all geometry information needed.
        for model in self.models.iter() {
            let triangles = vk::AccelerationStructureGeometryTrianglesDataKHRBuilder::new()
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: model.index_buffer.device_address(),
                })
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: model.vertex_buffer.device_address(),
                })
                .max_vertex(model.max_vertex_index)
                .vertex_stride(std::mem::size_of::<Vertex>() as u64)
                .transform_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: model.transform_buffer.device_address(),
                });

            let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                triangles: triangles.build(),
            };

            let geometry = vk::AccelerationStructureGeometryKHRBuilder::new()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES_KHR)
                .geometry(geometry_data)
                .flags(vk::GeometryFlagsKHR::OPAQUE_KHR);

            geometries.push(geometry);
        }

        // Iterate through all models and create their BLAS, scratchpads and geometry information.
        for (id, model) in self.models.iter().enumerate() {
            let query_info = vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
                .flags(
                    vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION_KHR
                        | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR,
                )
                .geometries(&geometries[id..id + 1])
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD_KHR)
                ._type(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL_KHR);

            let size_info = self.device.acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::HOST_KHR,
                &query_info,
                &[model.triangle_count],
            );

            let buffer = self.device.create_buffer(&BufferDescriptor {
                name: "BLAS Model Buffer",
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_location: vk_alloc::MemoryLocation::GpuOnly,
                sharing_mode: Default::default(),
                queues: Default::default(),
                size: size_info.acceleration_structure_size,
                flags: None,
            })?;

            let scratch_pad = self.device.create_buffer(&BufferDescriptor {
                name: "AS Scratchpad",
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_location: vk_alloc::MemoryLocation::GpuOnly,
                sharing_mode: Default::default(),
                queues: Default::default(),
                size: size_info.build_scratch_size,
                flags: None,
            })?;
            let scratch_pad_device_address = scratch_pad.device_address();
            scratch_pads.push(scratch_pad);

            let creation_info = vk::AccelerationStructureCreateInfoKHRBuilder::new()
                .buffer(buffer.raw)
                ._type(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL_KHR);

            let structure = self
                .device
                .create_acceleration_structure("RT Acceleration Structure", &creation_info)?;

            let blas = BLAS { structure, buffer };

            let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
                .flags(
                    vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION_KHR
                        | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR,
                )
                ._type(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL_KHR)
                .geometries(&geometries[id..id + 1])
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD_KHR)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_pad_device_address,
                })
                .dst_acceleration_structure(blas.structure.raw);

            let range = vk::AccelerationStructureBuildRangeInfoKHRBuilder::new()
                .primitive_count(model.triangle_count)
                .primitive_offset(0)
                .transform_offset(0)
                .first_vertex(0);

            infos.push(geometry_info);
            ranges.push(range.build());
            blas_vec.push(blas);
        }

        // Build all BLAS on the device using the compute queue.
        let compute_buffer = self.compute_pool.create_command_buffer(
            &self.timeline,
            self.timeline_value,
            self.timeline_value + 1,
        )?;
        self.timeline_value += 1;

        {
            let encoder = compute_buffer.record()?;
            encoder.build_acceleration_structures(&infos, &ranges);
        }
        self.compute_queue.submit(&compute_buffer)?;

        // We need to wait, or else the scratch pads will get deleted before the building of the AS is finished.
        self.timeline.wait_for_value(self.timeline_value)?;
        self.compute_pool.reset()?;

        Ok(())
    }
}

struct BLAS {
    structure: asche::AccelerationStructure,
    buffer: asche::Buffer,
}

struct Texture {
    view: asche::ImageView,
    _image: asche::Image,
}

impl Texture {
    fn create_offscreen_attachment(device: &asche::Device, extent: vk::Extent2D) -> Result<Self> {
        let image = device.create_image(&asche::ImageDescriptor {
            name: "Offscreen Texture",
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queues: vk::QueueFlags::GRAPHICS,
            image_type: vk::ImageType::_2D,
            format: OFFSCREEN_FORMAT,
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
        })?;

        let view = device.create_image_view(&asche::ImageViewDescriptor {
            name: "Offscreen Texture View",
            image: &image,
            view_type: vk::ImageViewType::_2D,
            format: OFFSCREEN_FORMAT,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            flags: None,
        })?;

        Ok(Texture {
            view,
            _image: image,
        })
    }
}

/// Each model has exactly one instance for this simple example.
struct Model {
    triangle_count: u32,
    max_vertex_index: u32,
    tlas_index: u32,
    blas_index: u32,
    transform_buffer: asche::Buffer,
    vertex_buffer: asche::Buffer,
    index_buffer: asche::Buffer,
}
