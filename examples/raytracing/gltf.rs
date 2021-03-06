use bytemuck::{Pod, Zeroable};
use glam::Mat4;

#[derive(Debug)]
pub struct Mesh {
    pub model_matrix: Mat4,
    pub material: usize,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

#[derive(Debug)]
pub struct Material {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
}

unsafe impl Pod for Vertex {}

unsafe impl Zeroable for Vertex {}

pub fn load_models(data: &[u8]) -> (Vec<Material>, Vec<Mesh>) {
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
                albedo,
                metallic,
                roughness: rough,
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
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| panic!("can't read positions"))
                .unwrap()
                .collect();

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .ok_or_else(|| panic!("can't read normals"))
                .unwrap()
                .collect();

            let tangents: Vec<[f32; 4]> = reader
                .read_tangents()
                .ok_or_else(|| panic!("can't read tangents"))
                .unwrap()
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

            Mesh {
                model_matrix: Mat4::from_cols_array_2d(&node.transform().matrix()),
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
