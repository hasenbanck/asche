use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};
use erupt::{vk, ExtendableFrom};
#[cfg(feature = "tracing")]
use tracing::info;
use ultraviolet::{Mat4, Vec3, Vec4};

use crate::gltf::{Material, Mesh, Vertex};
use crate::uploader::Uploader;

mod gltf;
mod uploader;

type Result<T> = std::result::Result<T, asche::AscheError>;

const SURFACE_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const OFFSCREEN_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

#[inline]
fn align_up(offset: u32, alignment: u32) -> u32 {
    (offset + (alignment - 1u32)) & !(alignment - 1u32)
}

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
            //  shaderStorageBufferArrayNonUniformIndexing
            features_v1_2: Some(
                vk::PhysicalDeviceVulkan12FeaturesBuilder::new()
                    .timeline_semaphore(true)
                    .buffer_device_address(true)
                    .scalar_block_layout(true)
                    .uniform_buffer_standard_layout(true)
                    .descriptor_indexing(true)
                    .descriptor_binding_partially_bound(true)
                    .descriptor_binding_variable_descriptor_count(true)
                    .runtime_descriptor_array(true)
                    .shader_storage_buffer_array_non_uniform_indexing(true),
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
    app.upload_uniforms()?;
    app.upload_model(&materials, &meshes)?;
    app.update_descriptor_sets();

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

struct RayTracingApplication {
    uniforms: Vec<asche::Buffer>,
    sbt_stride_addresses: Vec<vk::StridedDeviceAddressRegionKHR>,
    _sbt: asche::Buffer,
    tlas: Vec<TLAS>,
    blas: Vec<BLAS>,
    models: Vec<Model>,
    extent: vk::Extent2D,
    timeline: asche::TimelineSemaphore,
    timeline_value: u64,
    offscreen_attachment: Texture,
    renderpass: asche::RenderPass,
    postprocess_pipeline_layout: asche::PipelineLayout,
    postprocess_pipeline: asche::GraphicsPipeline,
    _postprocess_descriptor_pool: asche::DescriptorPool,
    _postprocess_descriptor_set_layout: asche::DescriptorSetLayout,
    postprocess_descriptor_set: asche::DescriptorSet,
    raytracing_pipeline_layout: asche::PipelineLayout,
    raytracing_pipeline: asche::RayTracingPipeline,
    _vertex_descriptor_set_layout: asche::DescriptorSetLayout,
    vertex_descriptor_set: asche::DescriptorSet,
    _index_descriptor_set_layout: asche::DescriptorSetLayout,
    index_descriptor_set: asche::DescriptorSet,
    _storage_descriptor_pool: asche::DescriptorPool,
    _raytracing_descriptor_pool: asche::DescriptorPool,
    _raytracing_descriptor_set_layout: asche::DescriptorSetLayout,
    raytracing_descriptor_set: asche::DescriptorSet,
    sampler: asche::Sampler,
    uploader: Uploader,
    graphics_pool: asche::GraphicsCommandPool,
    compute_pool: asche::ComputeCommandPool,
    graphics_queue: asche::GraphicsQueue,
    compute_queue: asche::ComputeQueue,
    device: asche::Device,
}

impl RayTracingApplication {
    fn new(
        mut device: asche::Device,
        mut compute_queue: asche::ComputeQueue,
        mut graphics_queue: asche::GraphicsQueue,
        transfer_queue: asche::TransferQueue,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let extent = vk::Extent2D { width, height };

        // Offscreen Attachment
        let offscreen_attachment = Texture::create_offscreen_attachment(&device, extent)?;

        // Sampler
        let sampler = device.create_sampler(&asche::SamplerDescriptor {
            name: "Offscreen Texture Sampler",
            ..Default::default()
        })?;

        // Utility
        let timeline_value = 0;
        let timeline = device.create_timeline_semaphore("Render Timeline", timeline_value)?;

        let compute_pool = compute_queue.create_command_pool()?;
        let graphics_pool = graphics_queue.create_command_pool()?;

        let mut uploader = Uploader::new(&device, transfer_queue)?;

        // Render pass
        let (
            renderpass,
            postprocess_descriptor_pool,
            postprocess_descriptor_set_layout,
            postprocess_descriptor_set,
            postprocess_pipeline_layout,
            postprocess_pipeline,
        ) = RayTracingApplication::create_postprocess_pipeline(&mut device)?;

        let (
            raytracing_descriptor_pool,
            raytracing_descriptor_set_layout,
            raytracing_descriptor_set,
            storage_descriptor_pool,
            vertex_descriptor_set_layout,
            vertex_descriptor_set,
            index_descriptor_set_layout,
            index_descriptor_set,
            raytracing_pipeline_layout,
            raytracing_pipeline,
            sbt,
            sbt_stride_addresses,
        ) = RayTracingApplication::create_rt_pipeline(&mut device, &mut uploader)?;

        Ok(Self {
            uniforms: vec![],
            sbt_stride_addresses,
            _sbt: sbt,
            tlas: vec![],
            blas: vec![],
            models: vec![],
            extent,
            timeline,
            timeline_value,
            offscreen_attachment,
            postprocess_pipeline,
            postprocess_pipeline_layout,
            _postprocess_descriptor_pool: postprocess_descriptor_pool,
            _postprocess_descriptor_set_layout: postprocess_descriptor_set_layout,
            postprocess_descriptor_set,
            renderpass,
            raytracing_pipeline,
            _vertex_descriptor_set_layout: vertex_descriptor_set_layout,
            vertex_descriptor_set,
            _index_descriptor_set_layout: index_descriptor_set_layout,
            index_descriptor_set,
            _storage_descriptor_pool: storage_descriptor_pool,
            raytracing_pipeline_layout,
            _raytracing_descriptor_pool: raytracing_descriptor_pool,
            _raytracing_descriptor_set_layout: raytracing_descriptor_set_layout,
            raytracing_descriptor_set,
            compute_pool,
            graphics_pool,
            sampler,
            uploader,
            compute_queue,
            graphics_queue,
            device,
        })
    }

    #[allow(clippy::type_complexity)]
    fn create_rt_pipeline(
        device: &mut asche::Device,
        uploader: &mut Uploader,
    ) -> Result<(
        asche::DescriptorPool,
        asche::DescriptorSetLayout,
        asche::DescriptorSet,
        asche::DescriptorPool,
        asche::DescriptorSetLayout,
        asche::DescriptorSet,
        asche::DescriptorSetLayout,
        asche::DescriptorSet,
        asche::PipelineLayout,
        asche::RayTracingPipeline,
        asche::Buffer,
        Vec<vk::StridedDeviceAddressRegionKHR>,
    )> {
        // Query for RT capabilities
        let mut raytrace_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHRBuilder::new();
        let properties =
            vk::PhysicalDeviceProperties2Builder::new().extend_from(&mut raytrace_properties);
        device.physical_device_properties(properties);

        // RT shader
        let mainfunctionname = std::ffi::CString::new("main").unwrap();
        let raygen_module = device.create_shader_module(
            "Raytrace Raygen Shader Module",
            include_bytes!("shader/raytrace.rgen.spv"),
        )?;
        let miss_module = device.create_shader_module(
            "Raytrace Miss Shader Module",
            include_bytes!("shader/raytrace.rmiss.spv"),
        )?;
        let close_hit_module = device.create_shader_module(
            "Raytrace Close Hit Shader Module",
            include_bytes!("shader/raytrace.rchit.spv"),
        )?;

        let raygen_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::RAYGEN_KHR)
            .module(raygen_module.raw)
            .name(&mainfunctionname);
        let miss_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::MISS_KHR)
            .module(miss_module.raw)
            .name(&mainfunctionname);
        let close_hit_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::CLOSEST_HIT_KHR)
            .module(close_hit_module.raw)
            .name(&mainfunctionname);

        // RT descriptor set layouts
        let bindings = [
            // TLAS
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // Camera Uniforms
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // Offscreen image
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(2)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // Light Uniforms
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(3)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(
                    vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::MISS_KHR,
                ),
            // Materials
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(4)
                .descriptor_count(1024)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        ];
        let flags = [
            vk::DescriptorBindingFlags::default(),
            vk::DescriptorBindingFlags::default(),
            vk::DescriptorBindingFlags::default(),
            vk::DescriptorBindingFlags::default(),
            vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        ];
        let mut layout_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
            .bindings(&bindings)
            .extend_from(&mut layout_flags);
        let raytracing_descriptor_set_layout =
            device.create_descriptor_set_layout("RT Descriptor Set Layout", layout_info)?;

        let vertex_bindings = [
            // Vertices
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(0)
                .descriptor_count(1024)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        ];
        let flags = [vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND];
        let mut layout_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
            .bindings(&vertex_bindings)
            .extend_from(&mut layout_flags);
        let vertex_descriptor_set_layout =
            device.create_descriptor_set_layout("Vertex Descriptor Set Layout", layout_info)?;

        let index_bindings = [
            // Indexes
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(0)
                .descriptor_count(1024)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        ];
        let flags = [vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND];
        let mut layout_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
            .bindings(&index_bindings)
            .extend_from(&mut layout_flags);
        let index_descriptor_set_layout =
            device.create_descriptor_set_layout("Index Descriptor Set Layout", layout_info)?;

        // RT descriptor pools + sets
        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(1)
                ._type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(1)
                ._type(vk::DescriptorType::UNIFORM_BUFFER),
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(1)
                ._type(vk::DescriptorType::STORAGE_IMAGE),
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(1)
                ._type(vk::DescriptorType::UNIFORM_BUFFER),
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(1024)
                ._type(vk::DescriptorType::STORAGE_BUFFER),
        ];
        let raytracing_descriptor_pool =
            device.create_descriptor_pool(&asche::DescriptorPoolDescriptor {
                name: "RT Descriptor Pool",
                max_sets: 2,
                pool_sizes: &pool_sizes,
                flags: None,
            })?;

        let raytracing_descriptor_set = raytracing_descriptor_pool.create_descriptor_set(
            "RT Descriptor Set",
            &raytracing_descriptor_set_layout,
            Some(1024),
        )?;

        let storage_pool_sizes = [vk::DescriptorPoolSizeBuilder::new()
            .descriptor_count(1024)
            ._type(vk::DescriptorType::STORAGE_BUFFER)];
        let storage_descriptor_pool =
            device.create_descriptor_pool(&asche::DescriptorPoolDescriptor {
                name: "Storage Descriptor Pool",
                max_sets: 4,
                pool_sizes: &storage_pool_sizes,
                flags: None,
            })?;

        let vertex_descriptor_set = storage_descriptor_pool.create_descriptor_set(
            "Vertex Descriptor Set",
            &vertex_descriptor_set_layout,
            Some(1024),
        )?;

        let index_descriptor_set = storage_descriptor_pool.create_descriptor_set(
            "Index Descriptor Set",
            &index_descriptor_set_layout,
            Some(1024),
        )?;

        // RT pipeline layout
        let layouts = [
            raytracing_descriptor_set_layout.raw,
            vertex_descriptor_set_layout.raw,
            index_descriptor_set_layout.raw,
        ];
        let layout_info = vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&layouts);
        let raytracing_pipeline_layout =
            device.create_pipeline_layout("RT Pipeline Layout", layout_info)?;

        // RT Pipeline
        let shader_stages = vec![raygen_stage, miss_stage, close_hit_stage];
        let max_bounce = raytrace_properties.max_ray_recursion_depth.min(1);

        let groups = [
            // Raygen
            vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
                ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR),
            // Miss
            vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
                ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR),
            // Hit
            vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
                ._type(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP_KHR)
                .general_shader(2)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
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

        let (sbt, sbt_stride_addresses) = RayTracingApplication::create_sbt(
            device,
            uploader,
            &raytrace_properties,
            &groups,
            &raytracing_pipeline,
        )?;

        Ok((
            raytracing_descriptor_pool,
            raytracing_descriptor_set_layout,
            raytracing_descriptor_set,
            storage_descriptor_pool,
            vertex_descriptor_set_layout,
            vertex_descriptor_set,
            index_descriptor_set_layout,
            index_descriptor_set,
            raytracing_pipeline_layout,
            raytracing_pipeline,
            sbt,
            sbt_stride_addresses,
        ))
    }

    fn create_sbt(
        device: &mut asche::Device,
        uploader: &mut Uploader,
        raytrace_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHRBuilder,
        groups: &[vk::RayTracingShaderGroupCreateInfoKHRBuilder],
        raytracing_pipeline: &asche::RayTracingPipeline,
    ) -> Result<(asche::Buffer, Vec<vk::StridedDeviceAddressRegionKHR>)> {
        let group_count = groups.len() as u32;
        let sbt_size = Self::calculate_sbt_size(&raytrace_properties, group_count);
        let mut sbt_data: Vec<u8> = vec![0; sbt_size as usize];
        device.ray_tracing_shader_group_handles(
            raytracing_pipeline.raw,
            0,
            group_count,
            sbt_data.as_mut_slice(),
        )?;

        let sbt = uploader.create_buffer_with_data(
            device,
            "SBT Buffer",
            cast_slice(sbt_data.as_slice()),
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS,
        )?;

        let group_size = Self::calculate_sbt_group_size(&raytrace_properties) as u64;
        let group_stride = group_size;
        let sbt_address = sbt.device_address();

        let sbt_stride_addresses = vec![
            // Raygen
            vk::StridedDeviceAddressRegionKHR {
                device_address: sbt_address,
                stride: group_stride,
                size: group_size,
            },
            // Miss
            vk::StridedDeviceAddressRegionKHR {
                device_address: sbt_address + group_size,
                stride: group_stride,
                size: group_size,
            },
            // Hit
            vk::StridedDeviceAddressRegionKHR {
                device_address: sbt_address + group_size * 2,
                stride: group_stride,
                size: group_size,
            },
            // Callable
            vk::StridedDeviceAddressRegionKHR::default(),
        ];

        Ok((sbt, sbt_stride_addresses))
    }

    fn calculate_sbt_group_size(
        raytrace_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHRBuilder,
    ) -> u32 {
        let group_handle_size = raytrace_properties.shader_group_handle_size;
        align_up(
            group_handle_size,
            raytrace_properties.shader_group_base_alignment,
        )
    }

    fn calculate_sbt_size(
        raytrace_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHRBuilder,
        group_count: u32,
    ) -> u32 {
        let group_size_aligned = Self::calculate_sbt_group_size(raytrace_properties);
        group_count * group_size_aligned
    }

    fn create_postprocess_pipeline(
        device: &mut asche::Device,
    ) -> Result<(
        asche::RenderPass,
        asche::DescriptorPool,
        asche::DescriptorSetLayout,
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
        let attachments = [vk::AttachmentDescription2Builder::new()
            .format(SURFACE_FORMAT)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .samples(vk::SampleCountFlagBits::_1)];

        let color_attachments = [vk::AttachmentReference2Builder::new()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let subpasses = [vk::SubpassDescription2Builder::new()
            .color_attachments(&color_attachments)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];

        let renderpass_info = vk::RenderPassCreateInfo2Builder::new()
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
            None,
        )?;

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
            postprocess_descriptor_set_layout,
            postprocess_descriptor_set,
            postprocess_pipeline_layout,
            postprocess_pipeline,
        ))
    }

    pub fn upload_uniforms(&mut self) -> Result<()> {
        let projection_matrix = ultraviolet::projection::rh_yup::perspective_reversed_infinite_z_vk(
            (70.0f32).to_radians(),
            self.extent.width as f32 / self.extent.height as f32,
            0.1,
        );
        let inv_projection_matrix = projection_matrix.inversed();
        let view_matrix = Mat4::look_at(Vec3::new(0.0, 2.0, 5.0), Vec3::zero(), Vec3::unit_y());
        let inv_view_matrix = view_matrix.inversed();
        let clear_color = Vec4::new(1.0, 1.0, 0.0, 1.0);
        let light_position = Vec4::new(-1.0, 1.0, 1.0, 1.0).normalized();
        let light_color = Vec4::new(1.0, 1.0, 1.0, 1.0);

        let camera_uniforms = CameraUniforms {
            view_matrix,
            projection_matrix,
            inv_view_matrix,
            inv_projection_matrix,
        };

        let light_uniforms = LightUniforms {
            clear_color,
            light_position,
            light_color,
        };

        let camera_uniforms_buffer = self.uploader.create_buffer_with_data(
            &self.device,
            "Camera Uniforms Buffer",
            cast_slice(&[camera_uniforms]),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS,
        )?;

        let light_uniforms_buffer = self.uploader.create_buffer_with_data(
            &self.device,
            "Lights Uniforms Buffer",
            cast_slice(&[light_uniforms]),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS,
        )?;

        self.uniforms.push(camera_uniforms_buffer);
        self.uniforms.push(light_uniforms_buffer);

        Ok(())
    }

    pub fn update_descriptor_sets(&self) {
        // Postprocessing Renderpass
        // Offscreen Image
        let image_info = [vk::DescriptorImageInfoBuilder::new()
            .sampler(self.sampler.raw)
            .image_view(self.offscreen_attachment.view.raw)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let postprocess_write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(self.postprocess_descriptor_set.raw)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info);

        // RT Renderpass
        // TLAS
        let structures: Vec<vk::AccelerationStructureKHR> =
            self.tlas.iter().map(|x| x.structure.raw).collect();

        let mut structure_info = vk::WriteDescriptorSetAccelerationStructureKHRBuilder::new()
            .acceleration_structures(&structures)
            .build();
        let mut tlas_write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(self.raytracing_descriptor_set.raw)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .extend_from(&mut structure_info);
        tlas_write.descriptor_count = 1;

        // Camera Uniform
        let buffer_info = [vk::DescriptorBufferInfoBuilder::new()
            .buffer(self.uniforms[0].raw)
            .offset(0)
            .range(std::mem::size_of::<CameraUniforms>() as u64)];
        let camera_write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(self.raytracing_descriptor_set.raw)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info);

        // Offscreen Image
        let image_info = [vk::DescriptorImageInfoBuilder::new()
            .image_view(self.offscreen_attachment.view.raw)
            .image_layout(vk::ImageLayout::GENERAL)];
        let offscreen_write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(self.raytracing_descriptor_set.raw)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info);

        // Light Uniform
        let buffer_info = [vk::DescriptorBufferInfoBuilder::new()
            .buffer(self.uniforms[1].raw)
            .offset(0)
            .range(std::mem::size_of::<LightUniforms>() as u64)];
        let light_write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(self.raytracing_descriptor_set.raw)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info);

        // Material, vertex and index buffer.
        let mut material_buffers: Vec<vk::DescriptorBufferInfoBuilder> =
            Vec::with_capacity(self.models.len());
        let mut vertex_buffers: Vec<vk::DescriptorBufferInfoBuilder> =
            Vec::with_capacity(self.models.len());
        let mut index_buffers: Vec<vk::DescriptorBufferInfoBuilder> =
            Vec::with_capacity(self.models.len());

        for model in self.models.iter() {
            let buffer_info = vk::DescriptorBufferInfoBuilder::new()
                .buffer(model.material_buffer.raw)
                .offset(0)
                .range(std::mem::size_of::<MaterialData>() as u64);
            material_buffers.push(buffer_info);

            let buffer_info = vk::DescriptorBufferInfoBuilder::new()
                .buffer(model.vertex_buffer.raw)
                .offset(0)
                .range(model.vertex_count as u64 * std::mem::size_of::<Vertex>() as u64);
            vertex_buffers.push(buffer_info);

            let buffer_info = vk::DescriptorBufferInfoBuilder::new()
                .buffer(model.index_buffer.raw)
                .offset(0)
                .range(model.index_count as u64 * std::mem::size_of::<u32>() as u64);
            index_buffers.push(buffer_info);
        }

        let material_write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(self.raytracing_descriptor_set.raw)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&material_buffers);

        let vertex_write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(self.vertex_descriptor_set.raw)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&vertex_buffers);

        let index_write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(self.index_descriptor_set.raw)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&index_buffers);

        self.device.update_descriptor_sets(
            &[
                postprocess_write,
                tlas_write,
                camera_write,
                offscreen_write,
                light_write,
                material_write,
                vertex_write,
                index_write,
            ],
            &[],
        );
    }
    /// Upload vertex data and prepares the TLAS and BLAS structures.
    pub fn upload_model(&mut self, materials: &[Material], meshes: &[Mesh]) -> Result<()> {
        for (id, mesh) in meshes.iter().enumerate() {
            let material = &materials[mesh.material];

            let material_data = MaterialData {
                model_matrix: mesh.model_matrix,
                albedo: material.albedo.into(),
                metallic: material.metallic,
                roughness: material.roughness,
            };

            let vertex_buffer = self.uploader.create_buffer_with_data(
                &self.device,
                &format!("Model {} Vertex Buffer", id),
                cast_slice(mesh.vertices.as_slice()),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            )?;
            let index_buffer = self.uploader.create_buffer_with_data(
                &self.device,
                &format!("Model {} Index Buffer", id),
                cast_slice(mesh.indices.as_slice()),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            )?;
            let material_buffer = self.uploader.create_buffer_with_data(
                &self.device,
                &format!("Model {} Material Buffer", id),
                cast_slice(&[material_data]),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            )?;

            self.models.push(Model {
                vertex_count: mesh.vertices.len() as u32,
                index_count: mesh.indices.len() as u32,
                material_buffer,
                vertex_buffer,
                index_buffer,
            })
        }

        self.init_blas()?;
        self.init_tlas()?;

        Ok(())
    }

    fn init_blas(&mut self) -> Result<()> {
        // We need to assemble all information that is needed to build the AS so that it outlives the loop were we assemble the command buffers.
        let mut scratchpad_size: u64 = 0;
        let mut max_sizes: Vec<u64> = Vec::with_capacity(self.models.len());
        let mut infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHRBuilder> =
            Vec::with_capacity(self.models.len());
        let mut ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> =
            Vec::with_capacity(self.models.len());

        // Create all geometry information needed.
        let geometries: Vec<vk::AccelerationStructureGeometryKHRBuilder> = self
            .models
            .iter()
            .map(|model| {
                let triangles = vk::AccelerationStructureGeometryTrianglesDataKHRBuilder::new()
                    .index_type(vk::IndexType::UINT32)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: model.index_buffer.device_address(),
                    })
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: model.vertex_buffer.device_address(),
                    })
                    .max_vertex(model.vertex_count)
                    .vertex_stride(std::mem::size_of::<Vertex>() as u64)
                    .transform_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: model.material_buffer.device_address(),
                    });

                let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                    triangles: triangles.build(),
                };

                vk::AccelerationStructureGeometryKHRBuilder::new()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES_KHR)
                    .geometry(geometry_data)
                    .flags(vk::GeometryFlagsKHR::OPAQUE_KHR)
            })
            .collect();

        // Calculate the maximal sizes of the AS and the scratch pad.
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
                vk::AccelerationStructureBuildTypeKHR::DEVICE_KHR,
                &query_info,
                &[model.index_count / 3],
            );

            max_sizes.push(size_info.acceleration_structure_size);

            if size_info.build_scratch_size > scratchpad_size {
                scratchpad_size = size_info.build_scratch_size
            }
        }

        // Create a scratch pad for the device to create the AS. We re-use it for each BLAS of a model.
        let scratch_pad = self.device.create_buffer(&asche::BufferDescriptor {
            name: "AS Scratchpad",
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            sharing_mode: Default::default(),
            queues: Default::default(),
            size: scratchpad_size,
            flags: None,
        })?;
        let scratch_pad_device_address = scratch_pad.device_address();

        // Create for each model a BLAS. We do this in one command buffer each, since a BLAS creation could take a long time
        // and this reduces the chance of timeouts and enabled to device to suspend the queue if needed (a device can only create
        // one AS at a time anyway).
        for (id, (model, size)) in self.models.iter().zip(&max_sizes).enumerate() {
            let blas = self.create_new_blas(&id, *size)?;
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
                .primitive_count(model.index_count / 3)
                .primitive_offset(0)
                .transform_offset(0)
                .first_vertex(0);

            infos.push(geometry_info);
            ranges.push(range.build());
            self.blas.push(blas);
        }

        let compacted_sizes = self.create_as_on_device(&infos, &ranges)?;
        self.compact_blas(&mut max_sizes, &compacted_sizes)?;

        self.compute_pool.reset()?;

        Ok(())
    }

    fn compact_blas(&mut self, max_sizes: &mut [u64], compacted_sizes: &[u64]) -> Result<()> {
        let command_buffer = self.compute_pool.create_command_buffer(
            &self.timeline,
            self.timeline_value,
            self.timeline_value + 1,
        )?;
        self.timeline_value += 1;

        let mut compacted_blas = Vec::with_capacity(self.blas.len());
        {
            let encoder = command_buffer.record()?;
            for ((id, blas), compacted) in self.blas.iter().enumerate().zip(compacted_sizes) {
                let compact_blas = self.create_new_blas(&id, *compacted)?;
                compacted_blas.push(compact_blas);

                let info = vk::CopyAccelerationStructureInfoKHRBuilder::new()
                    .mode(vk::CopyAccelerationStructureModeKHR::COMPACT_KHR)
                    .src(blas.structure.raw)
                    .dst(compacted_blas[compacted_blas.len() - 1].structure.raw);
                encoder.copy_acceleration_structure(&info);
            }
        }

        self.compute_queue.submit(&command_buffer)?;
        self.timeline.wait_for_value(self.timeline_value)?;

        self.blas = compacted_blas;

        #[cfg(feature = "tracing")]
        {
            let max_size: u64 = max_sizes.iter().sum();
            let compacted_size: u64 = compacted_sizes.iter().sum();
            info!(
                "Compacted BLAS from {} KiB to {} KiB",
                max_size / 1024,
                compacted_size / 1024
            );
        }

        Ok(())
    }

    fn create_new_blas(&self, id: &usize, compacted: u64) -> Result<BLAS> {
        let buffer = self.device.create_buffer(&asche::BufferDescriptor {
            name: &format!("Model {} BLAS Buffer", id),
            usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            sharing_mode: Default::default(),
            queues: Default::default(),
            size: compacted,
            flags: None,
        })?;

        let creation_info = vk::AccelerationStructureCreateInfoKHRBuilder::new()
            .buffer(buffer.raw)
            .size(compacted)
            ._type(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL_KHR);

        let structure = self
            .device
            .create_acceleration_structure("Model {} BLAS", &creation_info)?;

        Ok(BLAS {
            structure,
            _buffer: buffer,
        })
    }

    // Returns a query pool with the compacted sized.
    fn create_as_on_device(
        &mut self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHRBuilder],
        ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ) -> Result<Vec<u64>> {
        let query_pool = self.device.create_query_pool(
            "BLAS Compacted Size Query Pool",
            vk::QueryPoolCreateInfoBuilder::new()
                .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
                .query_count(self.models.len() as u32),
        )?;

        let mut command_buffers: Vec<asche::ComputeCommandBuffer> =
            Vec::with_capacity(self.models.len());
        for (id, _) in self.models.iter().enumerate() {
            let compute_buffer = self.compute_pool.create_command_buffer(
                &self.timeline,
                self.timeline_value,
                self.timeline_value + 1,
            )?;
            self.timeline_value += 1;

            {
                let encoder = compute_buffer.record()?;
                encoder.build_acceleration_structures(&infos[id..id + 1], &ranges[id..id + 1]);
                encoder.reset_query_pool(query_pool.raw, id as u32, 1);
                encoder.write_acceleration_structures_properties(
                    &[self.blas[id].structure.raw],
                    vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                    query_pool.raw,
                    id as u32,
                )
            }

            command_buffers.push(compute_buffer);
        }

        // Submit the command buffers and wait for them finishing.
        self.compute_queue.submit_all(&command_buffers)?;
        self.timeline.wait_for_value(self.timeline_value)?;

        // Get the compacted sizes
        let size = self.models.len();
        let mut compact_sizes = vec![0; size];
        query_pool.results(
            0,
            size as u32,
            cast_slice_mut(compact_sizes.as_mut_slice()),
            std::mem::size_of::<u64>() as u64,
            Some(vk::QueryResultFlags::WAIT),
        )?;

        Ok(compact_sizes)
    }

    fn init_tlas(&mut self) -> Result<()> {
        let instance_data: Vec<AccelerationStructureInstance> = self
            .blas
            .iter()
            .enumerate()
            .map(|(id, blas)| {
                #[rustfmt::skip]
                let matrix: [[f32; 4]; 3] = [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ];

                let address: vk::DeviceAddress = blas.structure.device_address();

                let mask = u32::MAX;
                let hit_group_id: u32 = 0;
                let flags: u32 =
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE_KHR.bits();

                // The ID is used to get the material later on (instance = model).
                let ici = (id as u32 & 0x00FFFFFF) | mask << 24;
                let isb = (hit_group_id & 0x00FFFFFF) | flags << 24;
                AccelerationStructureInstance {
                    transform: vk::TransformMatrixKHR { matrix },
                    instance_custom_index_and_mask: ici,
                    instance_shader_binding_table_record_offset_and_flags: isb,
                    acceleration_structure_reference: address,
                }
            })
            .collect();

        let instance_count = instance_data.len() as u32;

        let buffer = self.uploader.create_buffer_with_data(
            &self.device,
            "Model TLAS Instances",
            &cast_slice(instance_data.as_slice()),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
        )?;

        let geometry_instance_data =
            vk::AccelerationStructureGeometryInstancesDataKHRBuilder::new()
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: buffer.device_address(),
                })
                .array_of_pointers(false);

        let geometries = [vk::AccelerationStructureGeometryKHRBuilder::new()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES_KHR)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: geometry_instance_data.build(),
            })];

        // Get the build size
        let query_info = vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR)
            .geometries(&geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD_KHR)
            ._type(vk::AccelerationStructureTypeKHR::TOP_LEVEL_KHR);

        let size_info = self.device.acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE_KHR,
            &query_info,
            &[instance_count],
        );

        let buffer = self.device.create_buffer(&asche::BufferDescriptor {
            name: "Model TLAS Buffer",
            usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            size: size_info.acceleration_structure_size,
            flags: None,
        })?;

        let scratchpad = self.device.create_buffer(&asche::BufferDescriptor {
            name: "Model TLAS Scratchpad",
            usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queues: vk::QueueFlags::COMPUTE,
            size: size_info.build_scratch_size,
            flags: None,
        })?;

        let creation_info = vk::AccelerationStructureCreateInfoKHRBuilder::new()
            .buffer(buffer.raw)
            .size(size_info.acceleration_structure_size)
            ._type(vk::AccelerationStructureTypeKHR::TOP_LEVEL_KHR);

        let structure = self
            .device
            .create_acceleration_structure("Model TLAS", &creation_info)?;

        let geometry_infos = [vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR)
            .geometries(&geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD_KHR)
            ._type(vk::AccelerationStructureTypeKHR::TOP_LEVEL_KHR)
            .src_acceleration_structure(vk::AccelerationStructureKHR::null())
            .dst_acceleration_structure(structure.raw)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratchpad.device_address(),
            })];

        let ranges = [vk::AccelerationStructureBuildRangeInfoKHRBuilder::new()
            .primitive_count(instance_count)
            .primitive_offset(0)
            .transform_offset(0)
            .first_vertex(0)
            .build()];

        let compute_buffer = self.compute_pool.create_command_buffer(
            &self.timeline,
            self.timeline_value,
            self.timeline_value + 1,
        )?;
        self.timeline_value += 1;
        {
            let encoder = compute_buffer.record()?;
            encoder.build_acceleration_structures(&geometry_infos, &ranges);
        }

        self.compute_queue.submit(&compute_buffer)?;
        self.timeline.wait_for_value(self.timeline_value)?;

        self.tlas.push(TLAS {
            structure,
            _buffer: buffer,
        });

        self.compute_pool.reset()?;

        Ok(())
    }

    pub fn render(&mut self) -> Result<()> {
        let frame = self.device.get_next_frame()?;

        let command_buffer = self.graphics_pool.create_command_buffer(
            &self.timeline,
            Timeline::RenderStart.with_offset(self.timeline_value),
            Timeline::RenderEnd.with_offset(self.timeline_value),
        )?;

        {
            let encoder = command_buffer.record()?;
            encoder.set_viewport_and_scissor(
                vk::Rect2DBuilder::new()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(self.extent),
            );

            encoder.bind_raytrace_pipeline(&self.raytracing_pipeline);
            encoder.bind_descriptor_set(
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.raytracing_pipeline_layout.raw,
                0,
                &[self.raytracing_descriptor_set.raw],
                &[],
            );
            encoder.bind_descriptor_set(
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.raytracing_pipeline_layout.raw,
                1,
                &[self.vertex_descriptor_set.raw],
                &[],
            );
            encoder.bind_descriptor_set(
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.raytracing_pipeline_layout.raw,
                2,
                &[self.index_descriptor_set.raw],
                &[],
            );
            encoder.trace_rays_khr(
                &self.sbt_stride_addresses[0],
                &self.sbt_stride_addresses[1],
                &self.sbt_stride_addresses[2],
                &self.sbt_stride_addresses[3],
                self.extent.width,
                self.extent.height,
                1,
            );

            let pass = encoder.begin_render_pass(
                &self.renderpass,
                &[asche::RenderPassColorAttachmentDescriptor {
                    attachment: frame.view,
                    clear_value: Some(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [1.0, 0.0, 1.0, 1.0],
                        },
                    }),
                }],
                None,
                self.extent,
            )?;

            pass.bind_pipeline(&self.postprocess_pipeline);
            pass.bind_descriptor_sets(
                self.postprocess_pipeline_layout.raw,
                0,
                &[self.postprocess_descriptor_set.raw],
                &[],
            );

            // Draw the fullscreen triangle.
            pass.draw(3, 1, 0, 0);
        }

        self.graphics_queue.submit(&command_buffer)?;
        self.timeline
            .wait_for_value(Timeline::RenderEnd.with_offset(self.timeline_value))?;

        self.graphics_pool.reset()?;

        self.device.queue_frame(&self.graphics_queue, frame)?;
        self.timeline_value += Timeline::RenderEnd as u64;

        Ok(())
    }
}

// Hard copy with a workaround for following issue: https://gitlab.com/Friz64/erupt/-/issues/10
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AccelerationStructureInstance {
    pub transform: vk::TransformMatrixKHR,
    pub instance_custom_index_and_mask: u32,
    pub instance_shader_binding_table_record_offset_and_flags: u32,
    pub acceleration_structure_reference: u64,
}

unsafe impl Pod for AccelerationStructureInstance {}
unsafe impl Zeroable for AccelerationStructureInstance {}

struct TLAS {
    structure: asche::AccelerationStructure,
    _buffer: asche::Buffer,
}

struct BLAS {
    structure: asche::AccelerationStructure,
    _buffer: asche::Buffer,
}

struct Texture {
    view: asche::ImageView,
    _image: asche::Image,
}

impl Texture {
    fn create_offscreen_attachment(device: &asche::Device, extent: vk::Extent2D) -> Result<Self> {
        let image = device.create_image(&asche::ImageDescriptor {
            name: "Offscreen Texture",
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::INPUT_ATTACHMENT,
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
    index_count: u32,
    vertex_count: u32,
    material_buffer: asche::Buffer,
    vertex_buffer: asche::Buffer,
    index_buffer: asche::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraUniforms {
    view_matrix: Mat4,
    projection_matrix: Mat4,
    inv_view_matrix: Mat4,
    inv_projection_matrix: Mat4,
}

unsafe impl Pod for CameraUniforms {}

unsafe impl Zeroable for CameraUniforms {}

#[repr(C)]
#[derive(Clone, Copy)]
struct LightUniforms {
    clear_color: Vec4,
    light_position: Vec4,
    light_color: Vec4,
}

unsafe impl Pod for LightUniforms {}

unsafe impl Zeroable for LightUniforms {}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaterialData {
    model_matrix: [f32; 12],
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
}

unsafe impl Pod for MaterialData {}

unsafe impl Zeroable for MaterialData {}

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
