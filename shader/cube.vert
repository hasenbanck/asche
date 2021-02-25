#version 450

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_tex_coord;

layout (location = 0) out vec4 v_color;

layout (push_constant) uniform constants
{
    mat4 mvp;
} PC;

// TODO texture

void main() {
    gl_Position = PC.mvp * vec4(a_pos, 1.0);
    v_color = vec4(1.0, 1.0, 0.0, 1.0);
}

