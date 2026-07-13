use std::sync::Arc;
use wgpu::wgt::{CommandEncoderDescriptor, TextureViewDescriptor};
use wgpu::{include_wgsl, InstanceDescriptor, SurfaceConfiguration};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

struct State {
    window: Arc<Window>,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: wgpu::TextureFormat,
    surface: Option<wgpu::Surface<'static>>,
    render_pipeline: wgpu::RenderPipeline,
}

impl State {
    async fn new(event_loop: &ActiveEventLoop) -> anyhow::Result<Self> {
        let instance = pollster::block_on(wgpu::util::new_instance_with_webgpu_detection(
            InstanceDescriptor::new_with_display_handle_from_env(Box::new(
                event_loop.owned_display_handle(),
            )),
        ));

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

        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(surface_format.into())],
            }),
            multiview_mask: None,
            cache: None,
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

    fn resize(&self, width: u32, height: u32) {
        if let Some(surface) = &self.surface {
            surface.configure(
                &self.device,
                &SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: self.surface_format,
                    color_space: wgpu::SurfaceColorSpace::Auto,
                    width,
                    height,
                    present_mode: wgpu::PresentMode::AutoVsync,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![self.surface_format.add_srgb_suffix()],
                },
            );
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
                    ..Default::default()
                });
                renderpass.set_pipeline(&self.render_pipeline);
                renderpass.draw(0..3, 0..1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            self.window.pre_present_notify();
            self.queue.present(surface_texture);
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
                if let Some(state) = &self.state {
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
