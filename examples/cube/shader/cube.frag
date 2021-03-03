#version 450

layout (location = 0) in vec2 v_tex_coord;
layout (location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D albedo;

void main() {
    vec3 color = texture(albedo, v_tex_coord).rgb;
    f_color = vec4(color, 1.0f);
}
