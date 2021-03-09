#version 460

layout(location = 0) in vec2 v_tex_coord;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D t_offscreen;

void main() {
    f_color = pow(texture(t_offscreen, v_tex_coord).rgba, vec4(1.0 / 2.2));
}
