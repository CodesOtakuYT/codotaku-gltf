mod generated;

use crate::generated::shader_bindings;
use crate::generated::shader_bindings::shader::VertexInput;
use genmesh::generators::{IndexedPolygon, SharedVertex};
use genmesh::{generators, Triangulate, Vertices};
use std::f64::consts;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::{CommandEncoderDescriptor, TextureViewDescriptor};
use wgpu::{InstanceDescriptor, SurfaceConfiguration};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn camera_matrix(aspect_ratio: f64) -> glam::DMat4 {
    let projection =
        glam::dcamera::rh::proj::directx::perspective(consts::FRAC_PI_4, aspect_ratio, 1.0, 10.0);

    let view = glam::dcamera::rh::view::look_at_mat4(
        glam::DVec3::new(1.5, -5.0, 3.0),
        glam::DVec3::ZERO,
        glam::DVec3::Z,
    );

    projection * view
}

struct State {
    window: Arc<Window>,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: wgpu::TextureFormat,
    surface: Option<wgpu::Surface<'static>>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    depth_texture: Option<wgpu::TextureView>,
    index_count: u32,
    bind_group: shader_bindings::shader::WgpuBindGroup0,
}

impl State {
    async fn new(event_loop: &ActiveEventLoop) -> anyhow::Result<Self> {
        let instance = wgpu::util::new_instance_with_webgpu_detection(
            InstanceDescriptor::new_with_display_handle_from_env(Box::new(
                event_loop.owned_display_handle(),
            )),
        )
        .await;

        let window = Arc::new(event_loop.create_window(WindowAttributes::default())?);

        let temporary_surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&temporary_surface),
                ..Default::default()
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;

        let surface_capabilities = temporary_surface.get_capabilities(&adapter);
        let surface_format = surface_capabilities.formats[0];

        let (window_width, window_height): (u32, u32) = window.inner_size().into();
        let matrix = camera_matrix(window_width as f64 / window_height as f64).as_mat4();
        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(matrix.as_ref()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let depth_texture = create_depth_texture(&device, window_width, window_height);

        let pipeline_layout = shader_bindings::shader::create_pipeline_layout(&device);

        let bind_group = shader_bindings::shader::WgpuBindGroup0::from_bindings(
            &device,
            shader_bindings::shader::WgpuBindGroup0Entries::new(
                shader_bindings::shader::WgpuBindGroup0EntriesParams {
                    transform: wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: None,
                    },
                },
            ),
        );

        let shader = shader_bindings::shader::create_shader_module_embed_source(&device);
        let vertex_entry = shader_bindings::shader::vs_main_entry(wgpu::VertexStepMode::Vertex);
        let fragment_entry = shader_bindings::shader::fs_main_entry([Some(surface_format.into())]);

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: shader_bindings::shader::vertex_state(&shader, &vertex_entry),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(shader_bindings::shader::fragment_state(
                &shader,
                &fragment_entry,
            )),
            multiview_mask: None,
            cache: None,
        });

        let cube = generators::Cube::new();
        let vertices = cube
            .shared_vertex_iter()
            .map(|vertex| VertexInput {
                position: glam::Vec3::new(vertex.pos.x, vertex.pos.y, vertex.pos.z),
                normal: glam::Vec3::new(vertex.normal.x, vertex.normal.y, vertex.normal.z),
            })
            .collect::<Vec<_>>();
        let indices = cube
            .indexed_polygon_iter()
            .triangulate()
            .vertices()
            .map(|idx| idx as u16)
            .collect::<Vec<_>>();

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(Self {
            window,
            instance,
            adapter,
            device,
            queue,
            surface_format,
            surface: None,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            depth_texture: Some(depth_texture),
            index_count: indices.len() as u32,
            bind_group,
        })
    }

    fn resume(&mut self) -> anyhow::Result<()> {
        let surface = self.instance.create_surface(self.window.clone())?;
        let (width, height) = self.window.inner_size().into();
        self.resize(width, height);
        self.surface = Some(surface);
        self.window.request_redraw();
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) {
        if let Some(surface) = &self.surface {
            surface.configure(
                &self.device,
                &SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: self.surface_format,
                    width,
                    height,
                    present_mode: wgpu::PresentMode::AutoVsync,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![self.surface_format.add_srgb_suffix()],
                },
            );
            let (window_width, window_height): (u32, u32) = self.window.inner_size().into();
            let matrix = camera_matrix(window_width as f64 / window_height as f64).as_mat4();
            self.queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(matrix.as_ref()),
            );
            self.depth_texture = Some(create_depth_texture(&self.device, width, height));
        }
    }

    fn suspend(&mut self) {
        self.surface = None;
    }

    fn render(&mut self) -> anyhow::Result<()> {
        if let Some(surface) = &self.surface {
            let (window_width, window_height) = self.window.inner_size().into();

            let surface_texture = match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(texture) => texture,
                wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => {
                    return Ok(());
                }
                wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                    drop(texture);
                    self.resize(window_width, window_height);
                    return Ok(());
                }
                wgpu::CurrentSurfaceTexture::Outdated => {
                    self.resize(window_width, window_height);
                    return Ok(());
                }
                wgpu::CurrentSurfaceTexture::Validation => {
                    unreachable!("No error scope registered, so validation errors will panic")
                }
                wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface = Some(self.instance.create_surface(self.window.clone())?);
                    self.resize(window_width, window_height);
                    return Ok(());
                }
            };

            let surface_view = surface_texture.texture.create_view(&TextureViewDescriptor {
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());

            {
                let mut renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surface_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.depth_texture.as_ref().unwrap(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                renderpass.set_pipeline(&self.render_pipeline);
                renderpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                renderpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                self.bind_group.set(&mut renderpass);
                renderpass.draw_indexed(0..self.index_count, 0, 0..1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            self.window.pre_present_notify();
            surface_texture.present();
        }

        Ok(())
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.resume().unwrap();
        } else {
            let mut state = pollster::block_on(State::new(event_loop)).unwrap();
            state.resume().unwrap();
            self.state = Some(state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(inner_size) => {
                if let Some(state) = &mut self.state {
                    state.resize(inner_size.width, inner_size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    state.render().unwrap();
                    state.window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.suspend();
        }
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App::default())?;
    Ok(())
}
