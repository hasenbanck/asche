#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT hitPayload payload;

layout(set = 0, binding = 3, std430) uniform LightUniforms
{
    vec4 clear_color;
    vec4 proj;
    vec4 light_color;
}
lights;

void main()
{
    // BLUE
    payload.hit_value = vec4(0.0,0.0,1.0,1.0);
}