#version 460

layout (location = 0) out vec2 v_tex_coord;

void main() {
    // Draws a fullscreen triangle.
    v_tex_coord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(v_tex_coord * 2.0f - 1.0f, 1.0f, 1.0f);
}
