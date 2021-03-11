struct hitPayload
{
    vec4 hit_value;
};

struct Vertex
{
    vec3 position;
    vec3 normal;
    vec4 tangent;
};

struct Material
{
    vec4 albedo;
    float metallic;
    float roughness;
};
