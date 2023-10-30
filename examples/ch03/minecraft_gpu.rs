use bytemuck::{cast_slice, Pod, Zeroable};
use cgmath::Matrix4;
use std::{f32::consts::PI, iter};
use wgpu::{util::DeviceExt, VertexBufferLayout};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use wgpu_simplified as ws;
use wgpu_terrain::vertex_data as vd;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 4],
}

fn cube_vertices(side: f32) -> (Vec<Vertex>, Vec<u16>) {
    let (pos, _, _, _, ind, _) = vd::create_cube_data(side);
    let mut data: Vec<Vertex> = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        data.push(Vertex {
            position: [pos[i][0], pos[i][1], pos[i][2], 1.0],
        });
    }
    (data.to_vec(), ind)
}

struct State {
    init: ws::IWgpuInit,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    indices_len: u32,

    cs_pipeline: wgpu::ComputePipeline,
    cs_vertex_buffer: wgpu::Buffer,
    cs_uniform_buffer: wgpu::Buffer,
    cs_bind_group: wgpu::BindGroup,

    model_mat: Matrix4<f32>,
    view_mat: Matrix4<f32>,
    project_mat: Matrix4<f32>,
    msaa_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,

    update_buffers: bool,
    data_changed: bool,
    aspect_ratio: f32,
    animation_speed: f32,
    resolution: u32,
    water_level: f32,
    z_offset: f32,
    scale: f32,
    cube_side: f32,
    fps_counter: ws::FpsCounter,
}

impl State {
    async fn new(window: &Window, sample_count: u32, resolution: u32) -> Self {
        let init = ws::IWgpuInit::new(&window, sample_count, None).await;

        let resol = ws::round_to_multiple(resolution, 8);
        let vertices_count = resol * resol;
        println!("resolution = {}", resol);

        let shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("shader_minecraft.wgsl"));

        let cs_noise = include_str!("noise.wgsl");
        let cs_minecraft = include_str!("minecraft_comp.wgsl");
        let cs_combine = [cs_noise, cs_minecraft].join("\n");
        let cs_comp = init
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(cs_combine.into()),
            });

        // uniform data
        let model_mat = ws::create_model_mat(
            [-0.5 * resol as f32, -15.0, -0.5 * resol as f32],
            [0.0, PI / 15.0, 0.0],
            [1.5, 1.5, 1.5],
        );

        let camera_position = (40.0, 40.0, 50.0).into();
        let look_direction = (0.0, 0.0, 0.0).into();
        let up_direction = cgmath::Vector3::unit_y();

        let (view_mat, project_mat, vp_mat) = ws::create_vp_mat(
            camera_position,
            look_direction,
            up_direction,
            init.config.width as f32 / init.config.height as f32,
        );

        let mvp_mat = vp_mat * model_mat;

        // create vertex uniform buffers
        let vert_uniform_buffer =
            init.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Uniform Buffer"),
                    contents: cast_slice(mvp_mat.as_ref() as &[f32; 16]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // uniform bind group for vertex shader
        let (vert_bind_group_layout, vert_bind_group) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::VERTEX],
            &[vert_uniform_buffer.as_entire_binding()],
        );

        let vertex_buffer_layout = [
            VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4], // cube position
            },
            VertexBufferLayout {
                array_stride: 32,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4], // instance pos, col
            },
        ];

        let pipeline_layout = init
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&vert_bind_group_layout],
                push_constant_ranges: &[],
            });

        let mut ppl = ws::IRenderPipeline {
            shader: Some(&shader),
            pipeline_layout: Some(&pipeline_layout),
            vertex_buffer_layout: &vertex_buffer_layout,
            ..Default::default()
        };
        let pipeline = ppl.new(&init);

        let msaa_texture_view = ws::create_msaa_texture_view(&init);
        let depth_texture_view = ws::create_depth_view(&init);

        let (cube_vertex_data, cube_index_data) = cube_vertices(2.0);
        let cube_vertex_buffer =
            init.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Cube Vertex Buffer"),
                    contents: cast_slice(&cube_vertex_data),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        let cube_index_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cube Index Buffer"),
                contents: cast_slice(&cube_index_data),
                usage: wgpu::BufferUsages::INDEX,
            });

        // create compute pipeline for terrain
        let cs_vertex_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: 64 * vertices_count as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = [resol as f32, 5.0, 0.5, 2.0, 0.0, 0.0, 50.0, 0.2, 50.0];
        let cs_vertex_uniform_buffer =
            init.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Uniform Buffer"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let (cs_vertex_bind_group_layout, cs_vertex_bind_group) = ws::create_bind_group_storage(
            &init.device,
            vec![wgpu::ShaderStages::COMPUTE, wgpu::ShaderStages::COMPUTE],
            vec![
                wgpu::BufferBindingType::Storage { read_only: false },
                wgpu::BufferBindingType::Uniform,
            ],
            &[
                cs_vertex_buffer.as_entire_binding(),
                cs_vertex_uniform_buffer.as_entire_binding(),
            ],
        );

        let cs_pipeline_layout =
            init.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[&cs_vertex_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let cs_pipeline = init
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&cs_pipeline_layout),
                module: &cs_comp,
                entry_point: "cs_main",
            });

        Self {
            init,
            pipeline,
            vertex_buffer: cube_vertex_buffer,
            index_buffer: cube_index_buffer,
            uniform_bind_group: vert_bind_group,
            uniform_buffer: vert_uniform_buffer,
            indices_len: cube_index_data.len() as u32,

            cs_pipeline,
            cs_vertex_buffer,
            cs_uniform_buffer: cs_vertex_uniform_buffer,
            cs_bind_group: cs_vertex_bind_group,

            model_mat,
            view_mat,
            project_mat,
            msaa_texture_view,
            depth_texture_view,

            update_buffers: false,
            data_changed: false,
            aspect_ratio: 50.0,
            resolution: resol,
            animation_speed: 1.0,
            water_level: 0.2,
            z_offset: 0.0,
            scale: 50.0,
            cube_side: 2.0,
            fps_counter: ws::FpsCounter::default(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.init.size = new_size;
            self.init.config.width = new_size.width;
            self.init.config.height = new_size.height;
            self.init
                .surface
                .configure(&self.init.device, &self.init.config);

            self.project_mat =
                ws::create_projection_mat(new_size.width as f32 / new_size.height as f32, true);
            let mvp_mat = self.project_mat * self.view_mat * self.model_mat;
            self.init.queue.write_buffer(
                &self.uniform_buffer,
                0,
                cast_slice(mvp_mat.as_ref() as &[f32; 16]),
            );

            self.depth_texture_view = ws::create_depth_view(&self.init);
            if self.init.sample_count > 1 {
                self.msaa_texture_view = ws::create_msaa_texture_view(&self.init);
            }
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(keycode),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match keycode {
                VirtualKeyCode::Q => {
                    self.scale += 1.0;
                    self.update_buffers = true;
                    println!("scale = {}", self.scale);
                    true
                }
                VirtualKeyCode::A => {
                    self.scale -= 1.0;
                    if self.scale < 1.0 {
                        self.scale = 1.0;
                    }
                    self.update_buffers = true;
                    println!("scale = {}", self.scale);
                    true
                }
                VirtualKeyCode::W => {
                    self.aspect_ratio += 1.0;
                    self.update_buffers = true;
                    println!("aspect_ratio = {}", self.aspect_ratio);
                    true
                }
                VirtualKeyCode::S => {
                    self.aspect_ratio -= 1.0;
                    self.update_buffers = true;
                    println!("aspect_ratio = {}", self.aspect_ratio);
                    true
                }
                VirtualKeyCode::E => {
                    self.water_level += 0.01;
                    self.update_buffers = true;
                    println!("water_level = {}", self.water_level);
                    true
                }
                VirtualKeyCode::D => {
                    self.water_level -= 0.01;
                    self.update_buffers = true;
                    println!("water_level = {}", self.water_level);
                    true
                }
                VirtualKeyCode::R => {
                    self.cube_side += 0.1;
                    self.data_changed = true;
                    println!("cube_side = {}", self.cube_side);
                    true
                }
                VirtualKeyCode::F => {
                    self.cube_side -= 0.1;
                    if self.cube_side < 0.2 {
                        self.cube_side = 0.2;
                    }
                    self.data_changed = true;
                    println!("cube_side = {}", self.cube_side);
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        // update buffers:
        if self.update_buffers {
            let mvp_mat = self.project_mat * self.view_mat * self.model_mat;
            self.init.queue.write_buffer(
                &self.uniform_buffer,
                0,
                cast_slice(mvp_mat.as_ref() as &[f32; 16]),
            );
            self.update_buffers = false;
        }

        self.z_offset = 20.0 * self.animation_speed * dt.as_secs_f32();
        let params = [
            self.resolution as f32,
            5.0,
            0.5,
            2.0,
            0.0,
            self.z_offset,
            self.scale,
            self.water_level,
            self.aspect_ratio,
        ];
        self.init
            .queue
            .write_buffer(&self.cs_uniform_buffer, 0, cast_slice(&params));

        if self.data_changed {
            let (cube_vertex_data, _) = cube_vertices(self.cube_side);
            self.init
                .queue
                .write_buffer(&self.vertex_buffer, 0, cast_slice(&cube_vertex_data));
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        //let output = self.init.surface.get_current_frame()?.output;
        let output = self.init.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.init
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        // compute pass for vertices
        {
            let mut cs_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            cs_pass.set_pipeline(&self.cs_pipeline);
            cs_pass.set_bind_group(0, &self.cs_bind_group, &[]);
            cs_pass.dispatch_workgroups(self.resolution / 8, self.resolution / 8, 1);
        }

        // render pass
        {
            let color_attach = ws::create_color_attachment(&view);
            let msaa_attach = ws::create_msaa_color_attachment(&view, &self.msaa_texture_view);
            let color_attachment = if self.init.sample_count == 1 {
                color_attach
            } else {
                msaa_attach
            };
            let depth_attachment = ws::create_depth_stencil_attachment(&self.depth_texture_view);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(depth_attachment),
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..)); // for cube
            render_pass.set_vertex_buffer(1, self.cs_vertex_buffer.slice(..)); // for instance
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.draw_indexed(0..self.indices_len, 0, 0..self.resolution * self.resolution);
        }
        self.fps_counter.print_fps(5);
        self.init.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    let mut sample_count = 1 as u32;
    let mut resolution = 512u32;
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        sample_count = args[1].parse::<u32>().unwrap();
    }
    if args.len() > 2 {
        resolution = args[2].parse::<u32>().unwrap();
    }

    env_logger::init();
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();
    window.set_title(&*format!("{}", "terrain_gpu"));

    let mut state = pollster::block_on(State::new(&window, sample_count, resolution));
    let render_start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(_) => {
            let now = std::time::Instant::now();
            let dt = now - render_start_time;
            state.update(dt);

            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.init.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}