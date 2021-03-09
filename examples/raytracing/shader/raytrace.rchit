#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT hitPayload payload;

layout(set = 0, binding = 0) uniform accelerationStructureEXT TLAS;
layout(set = 0, binding = 3, std430) uniform LightUniforms
{
    vec4 clear_color;
    vec4 light_position;
    vec4 light_color;
}
light;
layout(set = 0, binding = 4, scalar) buffer MaterialsBuffer { Material m; } materials[];
layout(set = 1, binding = 0, scalar) buffer VertexBuffer { Vertex v[]; } vertices[];
layout(set = 2, binding = 0, scalar) buffer IndexBuffer { uint i[]; } indices[];

void main()
{
    // The ID of the mesh we hit (the index could also hit to an actual instance index, we are
    // currently expecting exactly one instance for each index to make things easier).
    uint mesh_id = gl_InstanceCustomIndexEXT;
    Material material = materials[mesh_id].m;

    ivec3 idx = ivec3(indices[nonuniformEXT(mesh_id)].i[3 * gl_PrimitiveID],
    indices[nonuniformEXT(mesh_id)].i[3 * gl_PrimitiveID + 1],
    indices[nonuniformEXT(mesh_id)].i[3 * gl_PrimitiveID + 2]);

    Vertex v0 = vertices[nonuniformEXT(mesh_id)].v[idx.x];
    Vertex v1 = vertices[nonuniformEXT(mesh_id)].v[idx.y];
    Vertex v2 = vertices[nonuniformEXT(mesh_id)].v[idx.z];

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Computing the normal at the hit position.
    vec3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;

    // Transforming the normal to world space.
    normal = normalize(vec3(material.model_matrix * vec4(normal, 0.0)));

    // Computing the coordinates of the hit position.
    vec3 world_pos = v0.position * barycentrics.x + v1.position * barycentrics.y + v2.position * barycentrics.z;

    // Transforming the position to world space.
    world_pos = vec3(material.model_matrix * vec4(world_pos, 1.0));

    // Light calculation.
    vec3  L = normalize(light.light_position.xyz - vec3(0.0));
    float dot_normal_L = max(dot(normal, L), 0.2);

    payload.hit_value = material.albedo * dot_normal_L;
}