#version 450

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_tex_coord;

layout (location = 0) out vec4 v_color;

// TODO rewrite both shaders, so that we properly draw the cube.

// TODO MVP
// TODO texture

void main() {
    gl_Position = vec4(a_pos, 1.0);
    v_color = vec4(1.0, 1.0, 0.0, 1.0);
}

