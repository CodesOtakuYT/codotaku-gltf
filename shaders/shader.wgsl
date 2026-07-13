struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@group(0)
@binding(0)
var<uniform> transform: mat4x4<f32>;

@vertex
fn vs_main(
    model: VertexInput
) -> VertexOutput {
    var result: VertexOutput;
    result.position = transform * vec4<f32>(model.position, 1.0);
    result.normal = model.normal;
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.3, -0.8, 0.5));
    let diffuse = max(dot(vertex.normal, light), 0.2);
    return vec4<f32>(diffuse, 0.0, 0.0, 1.0);
}