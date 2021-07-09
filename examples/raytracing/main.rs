use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};
use erupt::{vk, ExtendableFromConst, ExtendableFromMut};
use glam::{Mat4, Vec3, Vec4};
#[cfg(feature = "tracing")]
use tracing1::info;

use asche::{CommandBufferSemaphore, CommonCommands, Queues};

use crate::gltf::{Material, Mesh, Vertex};
use crate::uploader::Uploader;

mod gltf;
mod uploader;

type Result<T> = std::result::Result<T, asche::AscheError>;

const SURFACE_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const OFFSCREEN_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

#[inline]
fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + (alignment - 1)) & !(alignment - 1)
}

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("asche - raytracing example")
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
            app_name: "raytracing example",
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

    let (device, swapchain, queues) = instance.request_device(asche::DeviceConfiguration {
        swapchain_format: SURFACE_FORMAT,
        extensions: vec![
            // For RT support
            vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            vk::KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
            vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            // For GLSL_EXT_debug_printf support
            vk::KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        ],
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

    let Queues {
        mut compute_queues,
        mut graphics_queues,
        mut transfer_queues,
    } = queues;

    let mut app = RayTracingApplication::new(
        device,
        swapchain,
        compute_queues.pop().unwrap(),
        graphics_queues.pop().unwrap(),
        transfer_queues.pop().unwrap(),
        window.inner_size().width,
        window.inner_size().height,
    )
    .unwrap();

    let (materials, meshes) = gltf::load_models(include_bytes!("model.glb"));
    app.upload_uniforms()?;
    app.upload_model(&materials, &meshes)?;
    app.update_descriptor_sets();

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
                app.render().unwrap();
            }
            _ => (),
        }
    });
}

struct RayTracingApplication {
    uniforms: Vec<asche::Buffer>,
    sbt_stride_addresses: Vec<vk::StridedDeviceAddressRegionKHR>,
    _sbt: asche::Buffer,
    tlas: Vec<Tlas>,
    blas: Vec<Blas>,
    models: Vec<Model>,
    extent: vk::Extent2D,
    render_fence: asche::Fence,
    render_semaphore: asche::BinarySemaphore,
    presentation_semaphore: asche::BinarySemaphore,
    transfer_timeline: asche::TimelineSemaphore,
    transfer_timeline_value: u64,
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
    swapchain: asche::Swapchain,
    device: asche::Device,
}

impl RayTracingApplication {
    fn new(
        device: asche::Device,
        swapchain: asche::Swapchain,
        mut compute_queue: asche::ComputeQueue,
        mut graphics_queue: asche::GraphicsQueue,
        transfer_queue: asche::TransferQueue,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let extent = vk::Extent2D { width, height };

        // Sampler
        let sampler = device.create_sampler(&asche::SamplerDescriptor {
            name: "Offscreen Texture Sampler",
            ..Default::default()
        })?;

        // Utility
        let mut timeline_value = 0;
        let timeline = device.create_timeline_semaphore("Transfer Timeline", timeline_value)?;

        let compute_pool = compute_queue.create_command_pool()?;
        let mut graphics_pool = graphics_queue.create_command_pool()?;

        let mut uploader = Uploader::new(&device, transfer_queue)?;

        // Render pass
        let (
            renderpass,
            postprocess_descriptor_pool,
            postprocess_descriptor_set_layout,
            postprocess_descriptor_set,
            postprocess_pipeline_layout,
            postprocess_pipeline,
        ) = Self::create_postprocess_pipeline(&device)?;

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
        ) = Self::create_rt_pipeline(&device, &mut uploader)?;

        // Offscreen Image
        let offscreen_attachment = Self::create_offscreen_image(
            &device,
            extent,
            &mut graphics_pool,
            &mut graphics_queue,
            &timeline,
            &mut timeline_value,
        )?;

        let render_fence = device.create_fence("Render Fence")?;
        let render_semaphore = device.create_binary_semaphore("Render Semaphore")?;
        let presentation_semaphore = device.create_binary_semaphore("Presentation Semaphore")?;

        Ok(Self {
            uniforms: vec![],
            sbt_stride_addresses,
            _sbt: sbt,
            tlas: vec![],
            blas: vec![],
            models: vec![],
            extent,
            render_fence,
            render_semaphore,
            presentation_semaphore,
            transfer_timeline: timeline,
            transfer_timeline_value: timeline_value,
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
            swapchain,
        })
    }

    #[allow(clippy::type_complexity)]
    fn create_rt_pipeline(
        device: &asche::Device,
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

        let shader_stages = vec![raygen_stage, miss_stage, close_hit_stage];

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
        let layout_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
            .bindings(&bindings)
            .extend_from(&layout_flags);
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
        let layout_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
            .bindings(&vertex_bindings)
            .extend_from(&layout_flags);
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
        let layout_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
            .bindings(&index_bindings)
            .extend_from(&layout_flags);
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
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR),
        ];

        let rt_pipeline_info = vk::RayTracingPipelineCreateInfoKHRBuilder::new()
            .stages(&shader_stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(max_bounce)
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
        device: &asche::Device,
        uploader: &mut Uploader,
        raytrace_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHRBuilder,
        groups: &[vk::RayTracingShaderGroupCreateInfoKHRBuilder],
        raytracing_pipeline: &asche::RayTracingPipeline,
    ) -> Result<(asche::Buffer, Vec<vk::StridedDeviceAddressRegionKHR>)> {
        // A SBT orders the shaders in 4 groups:
        // 1. RG
        // 2. Miss
        // 3. HG
        // 4. Callable
        //
        // Each group must be aligned by "shader_group_base_alignment".
        // Each group can contain multiple entries which were defined
        // in the pipeline layout.
        //
        // Those entries are rightly packed in the handle_data and need
        // to be copied at the right location inside the SBT.
        //
        // Example for a "shader_group_base_alignment" of 64 with a
        // shader_group_handle_size of 32:
        //
        // ----     ----     ----     ----
        //| RG |   | MS |   | HG |   | CL |
        // ----     ----     ----     ----
        // 0       64        128     192
        //
        let shader_group_handle_size = raytrace_properties.shader_group_handle_size;
        let shader_group_base_alignment = raytrace_properties.shader_group_base_alignment as usize;

        let group_count = groups.len() as u32;
        let handle_data_size = shader_group_handle_size * group_count;

        let mut handle_data: Vec<u8> = vec![0; handle_data_size as usize];
        device.ray_tracing_shader_group_handles(
            raytracing_pipeline.raw,
            0,
            group_count,
            handle_data.as_mut_slice(),
        )?;

        // We only have one shader in the first three groups.
        let rg_group_size = shader_group_handle_size as usize;
        let miss_group_size = shader_group_handle_size as usize;
        let hg_group_size = shader_group_handle_size as usize;

        let rg_group_offfset = 0;
        let miss_group_offfset = align_up(
            rg_group_offfset + rg_group_size,
            shader_group_base_alignment,
        );
        let hg_group_offfset = align_up(
            miss_group_offfset + miss_group_size,
            shader_group_base_alignment,
        );

        let sbt_size = hg_group_offfset + hg_group_size;
        let mut sbt_data: Vec<u8> = vec![0; sbt_size];

        sbt_data[rg_group_offfset..rg_group_offfset + rg_group_size]
            .clone_from_slice(&handle_data[0..rg_group_size]);
        sbt_data[miss_group_offfset..miss_group_offfset + miss_group_size]
            .clone_from_slice(&handle_data[rg_group_size..(rg_group_size + miss_group_size)]);
        sbt_data[hg_group_offfset..hg_group_offfset + hg_group_size].clone_from_slice(
            &handle_data[(rg_group_size + miss_group_size)
                ..(rg_group_size + miss_group_size + hg_group_size)],
        );

        let sbt = uploader.create_buffer_with_data(
            device,
            "SBT Buffer",
            cast_slice(sbt_data.as_slice()),
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS,
        )?;

        let sbt_address = sbt.device_address();
        let sbt_stride_addresses = vec![
            // RG
            vk::StridedDeviceAddressRegionKHR {
                device_address: sbt_address + rg_group_offfset as u64,
                stride: shader_group_handle_size as u64,
                size: rg_group_size as u64,
            },
            // MISS
            vk::StridedDeviceAddressRegionKHR {
                device_address: sbt_address + miss_group_offfset as u64,
                stride: shader_group_handle_size as u64,
                size: miss_group_size as u64,
            },
            // HG
            vk::StridedDeviceAddressRegionKHR {
                device_address: sbt_address + hg_group_offfset as u64,
                stride: shader_group_handle_size as u64,
                size: hg_group_size as u64,
            },
            // CALL
            vk::StridedDeviceAddressRegionKHR {
                device_address: vk::DeviceAddress::default(),
                stride: 0,
                size: 0,
            },
        ];

        Ok((sbt, sbt_stride_addresses))
    }

    fn create_postprocess_pipeline(
        device: &asche::Device,
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

    fn create_offscreen_image(
        device: &asche::Device,
        extent: vk::Extent2D,
        pool: &mut asche::GraphicsCommandPool,
        queue: &mut asche::GraphicsQueue,
        timeline: &asche::TimelineSemaphore,
        timeline_value: &mut u64,
    ) -> Result<Texture> {
        let image = device.create_image(&asche::ImageDescriptor {
            name: "Offscreen Image",
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
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
            name: "Offscreen Image View",
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

        *timeline_value += 1;
        let command_buffer = pool.create_command_buffer(
            &[],
            &[CommandBufferSemaphore::Timeline {
                semaphore: timeline,
                stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                value: *timeline_value,
            }],
        )?;
        {
            let encoder = command_buffer.record()?;
            let image_barrier = vk::ImageMemoryBarrier2KHRBuilder::new()
                .src_stage_mask(vk::PipelineStageFlags2KHR::NONE_KHR)
                .src_access_mask(vk::AccessFlags2KHR::NONE_KHR)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2KHR::SHADER_WRITE_KHR)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image.raw)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            encoder.pipeline_barrier2(
                &vk::DependencyInfoKHRBuilder::new().image_memory_barriers(&[image_barrier]),
            );
        }

        queue.submit(&command_buffer, None)?;
        timeline.wait_for_value(*timeline_value)?;

        Ok(Texture { view, image })
    }

    pub fn upload_uniforms(&mut self) -> Result<()> {
        let projection_matrix = perspective_infinite_reverse_rh_yup(
            (90.0f32).to_radians(),
            self.extent.width as f32 / self.extent.height as f32,
            0.1,
        );
        let inv_projection_matrix = projection_matrix.inverse();
        let view_matrix =
            Mat4::look_at_rh(Vec3::new(0.0, 3.0, 3.0), Vec3::new(0.0, 2.0, 0.0), Vec3::Y);
        let inv_view_matrix = view_matrix.inverse();
        let clear_color = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let light_position = Vec4::new(-1.0, 1.0, 1.0, 1.0).normalize();
        let light_color = Vec4::new(1.0, 1.0, 1.0, 1.0);

        let camera_uniforms = CameraUniforms {
            view_matrix: view_matrix.to_cols_array(),
            projection_matrix: projection_matrix.to_cols_array(),
            inv_view_matrix: inv_view_matrix.to_cols_array(),
            inv_projection_matrix: inv_projection_matrix.to_cols_array(),
        };

        let light_uniforms = LightUniforms {
            clear_color: clear_color.into(),
            light_position: light_position.into(),
            light_color: light_color.into(),
        };

        let camera_uniforms_buffer = self.uploader.create_buffer_with_data(
            &self.device,
            "Camera Uniforms Buffer",
            cast_slice(&[camera_uniforms]),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::QueueFlags::GRAPHICS,
        )?;

        let light_uniforms_buffer = self.uploader.create_buffer_with_data(
            &self.device,
            "Lights Uniforms Buffer",
            cast_slice(&[light_uniforms]),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::QueueFlags::GRAPHICS,
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

        let structure_info = vk::WriteDescriptorSetAccelerationStructureKHRBuilder::new()
            .acceleration_structures(&structures)
            .build();
        let mut tlas_write = vk::WriteDescriptorSetBuilder::new()
            .dst_set(self.raytracing_descriptor_set.raw)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .extend_from(&structure_info);
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
                albedo: material.albedo,
                metallic: material.metallic,
                roughness: material.roughness,
            };

            // Vulkan expects a row major 3x4 transform matrix.
            let row_major_matrix = mesh.model_matrix.transpose();
            let transform = vk::TransformMatrixKHR {
                matrix: [
                    row_major_matrix.x_axis.into(),
                    row_major_matrix.y_axis.into(),
                    row_major_matrix.z_axis.into(),
                ],
            };

            let material_buffer = self.uploader.create_buffer_with_data(
                &self.device,
                &format!("Model {} Material Buffer", id),
                cast_slice(&[material_data]),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            )?;
            let vertex_buffer = self.uploader.create_buffer_with_data(
                &self.device,
                &format!("Model {} Vertex Buffer", id),
                cast_slice(mesh.vertices.as_slice()),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            )?;
            let index_buffer = self.uploader.create_buffer_with_data(
                &self.device,
                &format!("Model {} Index Buffer", id),
                cast_slice(mesh.indices.as_slice()),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            )?;
            let transform_buffer = self.uploader.create_buffer_with_data(
                &self.device,
                &format!("Model {} Transform Buffer", id),
                cast_slice(&transform.matrix),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            )?;

            self.models.push(Model {
                vertex_count: mesh.vertices.len() as u32,
                index_count: mesh.indices.len() as u32,
                material_buffer,
                vertex_buffer,
                index_buffer,
                transform_buffer,
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
                        device_address: model.transform_buffer.device_address(),
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

    #[allow(unused_variables)]
    fn compact_blas(&mut self, max_sizes: &mut [u64], compacted_sizes: &[u64]) -> Result<()> {
        self.transfer_timeline_value += 1;
        let command_buffer = self.compute_pool.create_command_buffer(
            &[],
            &[CommandBufferSemaphore::Timeline {
                semaphore: &self.transfer_timeline,
                stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                value: self.transfer_timeline_value,
            }],
        )?;

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

        self.compute_queue.submit(&command_buffer, None)?;
        self.transfer_timeline
            .wait_for_value(self.transfer_timeline_value)?;

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

    fn create_new_blas(&self, id: &usize, compacted: u64) -> Result<Blas> {
        let buffer = self.device.create_buffer(&asche::BufferDescriptor {
            name: &format!("Model {} BLAS Buffer", id),
            usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            memory_location: vk_alloc::MemoryLocation::GpuOnly,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queues: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
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

        Ok(Blas {
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
                &[CommandBufferSemaphore::Timeline {
                    semaphore: &self.transfer_timeline,
                    stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                    value: self.transfer_timeline_value,
                }],
                &[CommandBufferSemaphore::Timeline {
                    semaphore: &self.transfer_timeline,
                    stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                    value: self.transfer_timeline_value + 1,
                }],
            )?;
            self.transfer_timeline_value += 1;

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
        self.compute_queue.submit_all(&command_buffers, None)?;
        self.transfer_timeline
            .wait_for_value(self.transfer_timeline_value)?;

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
        let instance_data: Vec<vk::AccelerationStructureInstanceKHR> = self
            .blas
            .iter()
            .enumerate()
            .map(|(id, blas)| {
                // Vulkan expects a row major 3x4 transform matrix.
                let row_major_matrix = Mat4::IDENTITY.transpose();
                let transform = vk::TransformMatrixKHR {
                    matrix: [
                        row_major_matrix.x_axis.into(),
                        row_major_matrix.y_axis.into(),
                        row_major_matrix.z_axis.into(),
                    ],
                };

                vk::AccelerationStructureInstanceKHRBuilder::new()
                    .transform(transform)
                    .instance_custom_index(id as u32)
                    .mask(u32::MAX)
                    .instance_shader_binding_table_record_offset(0)
                    .flags(vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE_KHR)
                    .acceleration_structure_reference(blas.structure.device_address())
                    .build()
            })
            .collect();

        let instance_count = instance_data.len() as u32;

        // This buffer is only needed for creating the TLAS. Once it is crated, we can safely drop this.
        let instance_buffer = self.uploader.create_buffer_with_data(
            &self.device,
            "Model TLAS Instances",
            &cast_slice(instance_data.as_slice()),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
        )?;

        let geometry_instance_data =
            vk::AccelerationStructureGeometryInstancesDataKHRBuilder::new()
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: instance_buffer.device_address(),
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

        self.transfer_timeline_value += 1;
        let compute_buffer = self.compute_pool.create_command_buffer(
            &[],
            &[CommandBufferSemaphore::Timeline {
                semaphore: &self.transfer_timeline,
                stage: vk::PipelineStageFlags2KHR::NONE_KHR,
                value: self.transfer_timeline_value,
            }],
        )?;

        {
            let encoder = compute_buffer.record()?;
            encoder.build_acceleration_structures(&geometry_infos, &ranges);
        }

        self.compute_queue.submit(&compute_buffer, None)?;
        self.transfer_timeline
            .wait_for_value(self.transfer_timeline_value)?;

        self.tlas.push(Tlas {
            structure,
            _buffer: buffer,
        });

        self.compute_pool.reset()?;

        Ok(())
    }

    pub fn render(&mut self) -> Result<()> {
        let frame = self.swapchain.next_frame(&self.presentation_semaphore)?;

        let command_buffer = self.graphics_pool.create_command_buffer(
            &[],
            &[CommandBufferSemaphore::Binary {
                semaphore: &self.render_semaphore,
                stage: vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT_KHR,
            }],
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

            let image_barrier = vk::ImageMemoryBarrier2KHRBuilder::new()
                .src_stage_mask(vk::PipelineStageFlags2KHR::RAY_TRACING_SHADER_KHR)
                .src_access_mask(vk::AccessFlags2KHR::SHADER_WRITE_KHR)
                .old_layout(vk::ImageLayout::GENERAL)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::FRAGMENT_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2KHR::SHADER_READ_KHR)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.offscreen_attachment.image.raw)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            encoder.pipeline_barrier2(
                &vk::DependencyInfoKHRBuilder::new().image_memory_barriers(&[image_barrier]),
            );

            {
                let pass = encoder.begin_render_pass(
                    &mut self.swapchain,
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

            let image_barrier = vk::ImageMemoryBarrier2KHRBuilder::new()
                .src_stage_mask(vk::PipelineStageFlags2KHR::FRAGMENT_SHADER_KHR)
                .src_access_mask(vk::AccessFlags2KHR::SHADER_READ_KHR)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2KHR::SHADER_WRITE_KHR)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.offscreen_attachment.image.raw)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            encoder.pipeline_barrier2(
                &vk::DependencyInfoKHRBuilder::new().image_memory_barriers(&[image_barrier]),
            );
        }

        self.graphics_queue
            .submit(&command_buffer, Some(&self.render_fence))?;
        self.swapchain.queue_frame(
            &self.graphics_queue,
            frame,
            &[&self.presentation_semaphore, &self.render_semaphore],
        )?;

        self.render_fence.wait()?;
        self.render_fence.reset()?;

        self.graphics_pool.reset()?;

        Ok(())
    }
}

struct Tlas {
    structure: asche::AccelerationStructure,
    _buffer: asche::Buffer,
}

struct Blas {
    structure: asche::AccelerationStructure,
    _buffer: asche::Buffer,
}

struct Texture {
    view: asche::ImageView,
    image: asche::Image,
}

/// Each model has exactly one instance for this simple example.
struct Model {
    index_count: u32,
    vertex_count: u32,
    material_buffer: asche::Buffer,
    vertex_buffer: asche::Buffer,
    index_buffer: asche::Buffer,
    transform_buffer: asche::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CameraUniforms {
    view_matrix: [f32; 16],
    projection_matrix: [f32; 16],
    inv_view_matrix: [f32; 16],
    inv_projection_matrix: [f32; 16],
}

unsafe impl Pod for CameraUniforms {}

unsafe impl Zeroable for CameraUniforms {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct LightUniforms {
    clear_color: [f32; 4],
    light_position: [f32; 4],
    light_color: [f32; 4],
}

unsafe impl Pod for LightUniforms {}

unsafe impl Zeroable for LightUniforms {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct MaterialData {
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
}

unsafe impl Pod for MaterialData {}

unsafe impl Zeroable for MaterialData {}

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
