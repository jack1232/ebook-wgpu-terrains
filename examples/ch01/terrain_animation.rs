use bytemuck::cast_slice;
use cgmath::Matrix4;
use rand::Rng;
use std::{f32::consts::PI, iter, mem};
use wgpu::{util::DeviceExt, VertexBufferLayout};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use wgpu_simplified as ws;
use wgpu_terrain::terrain_data;

struct State {
    init: ws::IWgpuInit,
    pipelines: Vec<wgpu::RenderPipeline>,
    vertex_buffers: Vec<wgpu::Buffer>,
    index_buffers: Vec<wgpu::Buffer>,
    uniform_bind_groups: Vec<wgpu::BindGroup>,
    uniform_buffers: Vec<wgpu::Buffer>,
    model_mat: Matrix4<f32>,
    view_mat: Matrix4<f32>,
    project_mat: Matrix4<f32>,
    msaa_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    indices_lens: Vec<u32>,
    plot_type: u32,

    terrain: terrain_data::ITerrain,
    update_buffers: bool,
    aspect_ratio: f32,
    animation_speed: f32,
    fps_counter: ws::FpsCounter,
}

impl State {
    async fn new(window: &Window, sample_count: u32, width: u32, height: u32) -> Self {
        let init = ws::IWgpuInit::new(&window, sample_count, None).await;

        let shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("shader_unlit.wgsl"));

        // uniform data
        let model_mat = ws::create_model_mat(
            [-0.65 * width as f32, 5.0, -0.5 * height as f32],
            [0.0, PI / 15.0, 0.0],
            [1.0, 20.0, 1.0],
        );

        let camera_position = (40.0, 50.0, 60.0).into();
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
        let (vert_bind_group_layout2, vert_bind_group2) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::VERTEX],
            &[vert_uniform_buffer.as_entire_binding()],
        );

        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: mem::size_of::<terrain_data::Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3], // pos, col
        };

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
            vertex_buffer_layout: &[vertex_buffer_layout],
            ..Default::default()
        };

        let pipeline = ppl.new(&init);

        let vertex_buffer_layout2 = VertexBufferLayout {
            array_stride: mem::size_of::<terrain_data::Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3], // pos, col
        };

        let pipeline_layout2 =
            init.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout 2"),
                    bind_group_layouts: &[&vert_bind_group_layout2],
                    push_constant_ranges: &[],
                });

        let mut ppl2 = ws::IRenderPipeline {
            topology: wgpu::PrimitiveTopology::LineList,
            shader: Some(&shader),
            pipeline_layout: Some(&pipeline_layout2),
            vertex_buffer_layout: &[vertex_buffer_layout2],
            ..Default::default()
        };
        let pipeline2 = ppl2.new(&init);

        let msaa_texture_view = ws::create_msaa_texture_view(&init);
        let depth_texture_view = ws::create_depth_view(&init);

        let mut terrain = terrain_data::ITerrain::new();
        terrain.scale = 50.0;
        terrain.water_level = 0.3;
        terrain.width = width;
        terrain.height = height;
        let vertex_data = terrain.create_terrain_data_with_water_level();
        let index_data = terrain.create_indices(width, height);

        let vertex_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: cast_slice(&vertex_data.0),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        let vertex_buffer2 = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer 2"),
                contents: cast_slice(&vertex_data.1),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        let index_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&index_data.0),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });

        let index_buffer2 = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer 2"),
                contents: bytemuck::cast_slice(&index_data.1),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            init,
            pipelines: vec![pipeline, pipeline2],
            vertex_buffers: vec![vertex_buffer, vertex_buffer2],
            index_buffers: vec![index_buffer, index_buffer2],
            uniform_bind_groups: vec![vert_bind_group, vert_bind_group2],
            uniform_buffers: vec![vert_uniform_buffer],
            model_mat,
            view_mat,
            project_mat,
            msaa_texture_view,
            depth_texture_view,
            indices_lens: vec![index_data.0.len() as u32, index_data.1.len() as u32],
            plot_type: 0,

            terrain,
            update_buffers: false,
            aspect_ratio: 20.0,
            animation_speed: 1.0,
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
                &self.uniform_buffers[0],
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
                VirtualKeyCode::Space => {
                    self.plot_type = (self.plot_type + 1) % 3;
                    true
                }
                VirtualKeyCode::LControl => {
                    let mut rng = rand::thread_rng();
                    self.terrain.seed = rng.gen_range(1..65536);
                    self.update_buffers = true;
                    println!("seed = {}", self.terrain.seed);
                    true
                }
                VirtualKeyCode::Q => {
                    self.terrain.scale += 1.0;
                    self.update_buffers = true;
                    println!("scale = {}", self.terrain.scale);
                    true
                }
                VirtualKeyCode::A => {
                    self.terrain.scale -= 1.0;
                    if self.terrain.scale < 1.0 {
                        self.terrain.scale = 1.0;
                    }
                    self.update_buffers = true;
                    println!("scale = {}", self.terrain.scale);
                    true
                }
                VirtualKeyCode::W => {
                    self.terrain.octaves += 1;
                    self.update_buffers = true;
                    println!("octaves = {}", self.terrain.octaves);
                    true
                }
                VirtualKeyCode::S => {
                    self.terrain.octaves -= 1;
                    if self.terrain.octaves < 1 {
                        self.terrain.octaves = 1;
                    }
                    self.update_buffers = true;
                    println!("octaves = {}", self.terrain.octaves);
                    true
                }
                VirtualKeyCode::E => {
                    self.terrain.offsets[0] += 1.0;
                    self.update_buffers = true;
                    println!("offset_x = {}", self.terrain.offsets[0]);
                    true
                }
                VirtualKeyCode::D => {
                    self.terrain.offsets[0] -= 1.0;
                    self.update_buffers = true;
                    println!("offset_x = {}", self.terrain.offsets[0]);
                    true
                }
                VirtualKeyCode::R => {
                    self.terrain.offsets[1] += 1.0;
                    self.update_buffers = true;
                    println!("offset_z = {}", self.terrain.offsets[1]);
                    true
                }
                VirtualKeyCode::F => {
                    self.terrain.offsets[1] -= 1.0;
                    self.update_buffers = true;
                    println!("offset_z = {}", self.terrain.offsets[1]);
                    true
                }
                VirtualKeyCode::T => {
                    self.aspect_ratio += 1.0;
                    self.update_buffers = true;
                    println!("aspect_ratio = {}", self.aspect_ratio);
                    true
                }
                VirtualKeyCode::G => {
                    self.aspect_ratio -= 1.0;
                    self.update_buffers = true;
                    println!("aspect_ratio = {}", self.aspect_ratio);
                    true
                }
                VirtualKeyCode::Y => {
                    self.terrain.water_level += 0.01;
                    self.update_buffers = true;
                    println!("water_level = {}", self.terrain.water_level);
                    true
                }
                VirtualKeyCode::H => {
                    self.terrain.water_level -= 0.01;
                    self.update_buffers = true;
                    println!("water_level = {}", self.terrain.water_level);
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
            self.model_mat = ws::create_model_mat(
                [
                    -0.65 * self.terrain.width as f32,
                    5.0,
                    -0.5 * self.terrain.height as f32,
                ],
                [0.0, PI / 15.0, 0.0],
                [1.0, self.aspect_ratio, 1.0],
            );
            let mvp_mat = self.project_mat * self.view_mat * self.model_mat;
            self.init.queue.write_buffer(
                &self.uniform_buffers[0],
                0,
                cast_slice(mvp_mat.as_ref() as &[f32; 16]),
            );
            self.update_buffers = false;
        }

        self.terrain.offsets[1] = -20.0 * self.animation_speed * dt.as_secs_f32();
        let vertex_data = self.terrain.create_terrain_data_with_water_level();
        self.init
            .queue
            .write_buffer(&self.vertex_buffers[0], 0, cast_slice(&vertex_data.0));
        self.init
            .queue
            .write_buffer(&self.vertex_buffers[1], 0, cast_slice(&vertex_data.1));
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

            let plot_type = if self.plot_type == 1 {
                "shape_only"
            } else if self.plot_type == 2 {
                "wireframe_only"
            } else {
                "both"
            };

            if plot_type == "shape_only" || plot_type == "both" {
                render_pass.set_pipeline(&self.pipelines[0]);
                render_pass.set_vertex_buffer(0, self.vertex_buffers[0].slice(..));
                render_pass
                    .set_index_buffer(self.index_buffers[0].slice(..), wgpu::IndexFormat::Uint32);
                render_pass.set_bind_group(0, &self.uniform_bind_groups[0], &[]);
                render_pass.draw_indexed(0..self.indices_lens[0], 0, 0..1);
            }

            if plot_type == "wireframe_only" || plot_type == "both" {
                render_pass.set_pipeline(&self.pipelines[1]);
                render_pass.set_vertex_buffer(0, self.vertex_buffers[1].slice(..));
                render_pass
                    .set_index_buffer(self.index_buffers[1].slice(..), wgpu::IndexFormat::Uint32);
                render_pass.set_bind_group(0, &self.uniform_bind_groups[1], &[]);
                render_pass.draw_indexed(0..self.indices_lens[1], 0, 0..1);
            }
        }
        self.fps_counter.print_fps(5);
        self.init.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    let mut sample_count = 1 as u32;
    let mut width = 200u32;
    let mut height = 200u32;
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        sample_count = args[1].parse::<u32>().unwrap();
    }
    if args.len() > 2 {
        width = args[2].parse::<u32>().unwrap();
    }
    if args.len() > 3 {
        height = args[3].parse::<u32>().unwrap();
    }

    env_logger::init();
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();
    window.set_title(&*format!("{}", "terrain_animation"));

    let mut state = pollster::block_on(State::new(&window, sample_count, width, height));
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