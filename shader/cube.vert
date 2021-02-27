#version 450

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_tex_coord;

layout (location = 0) out vec2 v_tex_coord;

layout (push_constant) uniform constants
{
    mat4 mvp;
} PC;

void main() {
    gl_Position = PC.mvp * vec4(a_pos, 1.0);
    v_tex_coord = a_tex_coord;
}
