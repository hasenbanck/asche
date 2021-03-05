use bytemuck::{Pod, Zeroable};
use ultraviolet::{Mat4, Vec3, Vec4};

pub(crate) struct Mesh {
    /// Vulkan expects a 3x4 Row Major transform matrix.
    pub(crate) model_matrix: [f32; 12],
    pub(crate) material: usize,
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) indices: Vec<u32>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Material {
    pub(crate) albedo: Vec4,
    pub(crate) metallic: f32,
    pub(crate) rough: f32,
}

unsafe impl Pod for Material {}

unsafe impl Zeroable for Material {}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Vertex {
    pub(crate) position: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) tangent: Vec4,
}

unsafe impl Pod for Vertex {}

unsafe impl Zeroable for Vertex {}

pub(crate) fn load_models(data: &[u8]) -> (Vec<Material>, Vec<Mesh>) {
    let gltf = gltf::Gltf::from_slice(data).unwrap();
    let data = import_buffer_data(&gltf.document, gltf.blob);
    let document = gltf.document;

    let materials: Vec<Material> = document
        .materials()
        .map(|material| {
            let albedo = material.pbr_metallic_roughness().base_color_factor();
            let metallic = material.pbr_metallic_roughness().metallic_factor();
            let rough = material.pbr_metallic_roughness().roughness_factor();

            Material {
                albedo: Vec4::from(albedo),
                metallic,
                rough,
            }
        })
        .collect();

    let meshes: Vec<Mesh> = document
        .nodes()
        .filter(|node| node.mesh().is_some())
        .map(|node| {
            let mesh = node.mesh().unwrap();
            let primitive = mesh.primitives().next().unwrap();
            let reader = primitive.reader(|buffer| Some(&data[buffer.index()]));
            let positions: Vec<Vec3> = reader
                .read_positions()
                .ok_or_else(|| panic!("can't read positions"))
                .unwrap()
                .map(Vec3::from)
                .collect();

            let normals: Vec<Vec3> = reader
                .read_normals()
                .ok_or_else(|| panic!("can't read normals"))
                .unwrap()
                .map(Vec3::from)
                .collect();

            let tangents: Vec<Vec4> = reader
                .read_tangents()
                .ok_or_else(|| panic!("can't read tangents"))
                .unwrap()
                .map(Vec4::from)
                .collect();

            let indices: Vec<u32> = reader
                .read_indices()
                .ok_or_else(|| panic!("can't read indices"))
                .unwrap()
                .into_u32()
                .collect();

            let vertices: Vec<Vertex> = positions
                .iter()
                .zip(normals.iter())
                .zip(tangents.iter())
                .map(|((p, n), t)| Vertex {
                    position: (*p),
                    normal: (*n),
                    tangent: (*t),
                })
                .collect();

            let cols = Mat4::from(node.transform().matrix()).cols;

            // Ultraviolet is column major, vulkan expects row major
            #[rustfmt::skip]
                let model_matrix: [f32; 12] = [
                cols[0].x, cols[0].y, cols[0].z,
                cols[1].x, cols[1].y, cols[1].z,
                cols[2].x, cols[2].y, cols[2].z,
                cols[3].x, cols[3].y, cols[3].z,
            ];

            Mesh {
                model_matrix,
                material: primitive.material().index().unwrap(),
                vertices,
                indices,
            }
        })
        .collect();

    (materials, meshes)
}

/// Import the buffer data referenced by a glTF document.
fn import_buffer_data(document: &gltf::Document, mut blob: Option<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut buffers = Vec::new();
    for buffer in document.buffers() {
        let mut data = match buffer.source() {
            gltf::buffer::Source::Bin => blob.take().ok_or_else(|| panic!("can't take blob data")),
            _ => panic!("unsupported source for buffer data"),
        }
        .unwrap();
        if data.len() < buffer.length() {
            panic!("buffer wasn't fully filled");
        }
        while data.len() % 4 != 0 {
            data.push(0);
        }
        buffers.push(data);
    }
    buffers
}
