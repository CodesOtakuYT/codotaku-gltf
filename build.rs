use wgsl_bindgen::{GlamWgslTypeMap, WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    WgslBindgenOptionBuilder::default()
        .workspace_root("shaders")
        .add_entry_point("shaders/shader.wgsl")
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .type_map(GlamWgslTypeMap) // Use glam for math types
        .output("src/generated/shader_bindings.rs")
        .build()?
        .generate()?;
    Ok(())
}
