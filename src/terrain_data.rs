#![allow(dead_code)]
use noise::{NoiseFn, Perlin};
use super::colormap;
use bytemuck:: {Pod, Zeroable};
use wgpu_simplified as ws;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

pub struct ITerrain {
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    pub octaves: u32,
    pub persistence: f32,
    pub lacunarity: f32,
    pub offsets: [f32; 2],
    pub water_level: f32,
    pub scale: f32,
    pub colormap_name: String,
    pub wireframe_color: String,
    pub chunk_size: u32,
    pub level_of_detail: u32,
    pub normalize_mode: String,
}

impl Default for ITerrain {
    fn default() -> Self {
        Self {
            width: 200,
            height: 200,
            seed: 1232,
            octaves: 5,
            persistence: 0.5,
            lacunarity: 2.0,
            offsets: [0.0, 0.0],
            water_level: 0.0,
            scale: 10.0,
            colormap_name: "terrain".to_string(),
            wireframe_color: "white".to_string(),
            chunk_size: 241,
            level_of_detail: 0,
            normalize_mode: "local".to_string(),
        }
    }
}

impl ITerrain {
    pub fn new() -> Self {
        Default::default()
    }

    fn create_noise_map(&mut self, width: u32, height: u32) -> Vec<Vec<f32>>{
        let rng = ws::seed_random_number(self.seed as u64);
        let perlin = Perlin::new(self.seed);
       
        let mut offsets:Vec<[f32;2]> = vec![];
        for _i in 0..self.octaves {
            let offsetx = 100000f32 * (2.0 * rng - 1.0) + self.offsets[0];
            let offsetz = 100000f32 * (2.0 * rng - 1.0) + self.offsets[1];
            offsets.push([offsetx, offsetz]);
        }

        let mut noise_map:Vec<Vec<f32>> = vec![];
        let mut height_min = f32::MAX;
        let mut height_max = f32::MIN;
        let halfw = 0.5 * width as f32;
        let halfh = 0.5 * height as f32;

        for x in 0..width {
            let mut p1:Vec<f32> = vec![];
            for z in 0..height {
                let mut amplitude = 1f32;
                let mut frequency = 1f32;
                let mut noise_height = 0f32;

                for i in 0..self.octaves {
                    let sample_x = (x as f32 - halfw + offsets[i as usize][0]) * frequency / self.scale;
                    let sample_y = (z as f32 - halfh + offsets[i as usize][1]) * frequency / self.scale;
                    let y = perlin.get([sample_x as f64, sample_y as f64]) as f32;
                    noise_height += y * amplitude;
                    amplitude *= self.persistence;
                    frequency *= self.lacunarity;
                }
                height_min = if noise_height < height_min { noise_height } else { height_min };
                height_max = if noise_height > height_max { noise_height } else { height_max };
                p1.push(noise_height);
            }
            noise_map.push(p1);
        }

        if self.normalize_mode == "global" {
            height_min = -1.0;
            height_max = 1.0;
        }

        for x in 0..width as usize {
            for z in 0..height as usize {
                noise_map[x][z] = (noise_map[x][z] - height_min)/(height_max - height_min);
            }
        }

        noise_map
    }

    pub fn create_indices(&mut self, width: u32, height: u32) -> (Vec<u32>, Vec<u32>) {
        let n_vertices_per_row = height;
        let mut indices:Vec<u32> = vec![];
        let mut indices2:Vec<u32> = vec![];

        for i in 0..width - 1 {
            for j in 0..height - 1 {
                let idx0 = j + i * n_vertices_per_row;
                let idx1 = j + 1 + i * n_vertices_per_row;
                let idx2 = j + 1 + (i + 1) * n_vertices_per_row;
                let idx3 = j + (i + 1) * n_vertices_per_row;  
                indices.extend([idx0, idx1, idx2, idx2, idx3, idx0]);
                indices2.extend([idx0, idx1, idx0, idx3]);
                if i == width - 2 || j == height - 1 {
                    indices2.extend([idx1, idx2, idx2, idx3]);
                }
            }
        }
        (indices, indices2)
    }

    pub fn create_terrain_data(&mut self) -> (Vec<Vertex>, Vec<Vertex>) {
        let cdata = colormap::colormap_data(&self.colormap_name);
        let cdata2 = colormap::colormap_data(&self.wireframe_color);
        let noise_map = self.create_noise_map(self.width, self.height);

        let mut data:Vec<Vertex> = vec![];
        let mut data2:Vec<Vertex> = vec![];

        for x in 0..self.width as usize {
            for z in 0..self.height as usize {
                let y = if noise_map[x][z].is_finite() { noise_map[x][z] } else { 0.0 };

                let position = [x as f32, y, z as f32];
                let color = colormap::color_lerp(cdata, 0.0, 1.0, y);
                let color2 = colormap::color_lerp(cdata2, 0.0, 1.0, y);

                data.push(Vertex { position, color });
                data2.push(Vertex { position, color: color2 });
            }
        }
        (data, data2)        
    }

    fn terrian_colormap_data(&mut self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<f32>) {
        let cdata = vec![
            [0.055f32, 0.529, 0.8],
            [0.761, 0.698, 0.502],
            [0.204, 0.549, 0.192],
            [0.353, 0.302, 0.255],
            [1.0, 0.98, 0.98]
        ];
        let cdata2 = vec![[1f32, 1.0, 1.0]; 5];
        let ta = vec![0.0f32, 0.3, 0.35, 0.6, 0.9, 1.0];
        (cdata, cdata2, ta)
    }

    fn shift_water_level(&mut self, ta:&Vec<f32>) -> Vec<f32> {
        let mut t1 = vec![0f32; 6];
        let r = (1.0 - self.water_level)/(1.0 - ta[1]);
        t1[1] = self.water_level;
        for i in 1..5usize {
            let del = ta[i+1] - ta[i];
            t1[i+1] = t1[i] + r * del;
        }
        t1
    }

    fn color_lerp(&mut self, color:&Vec<[f32;3]>, ta:&Vec<f32>, t:f32) -> [f32;3] {
        let len = 6usize;
        let mut res = [0f32;3];
        for i in 0..len - 1 {
            if t >= ta[i] && t < ta[i + 1] {
                res = color[i];
            }
        }
        if t == ta[len-1] {
            res = color[len-2];
        }
        res
    }

    fn add_terrain_colors(&mut self, color:&Vec<[f32;3]>, ta:&Vec<f32>, tmin:f32, tmax:f32, t:f32) -> [f32;3] {
        let mut tt = if t < tmin { tmin } else if t > tmax { tmax } else { t };
        tt = (tt - tmin)/(tmax - tmin);
        let t1 = self.shift_water_level(ta);
        self.color_lerp(color, &t1, tt)
    }

    pub fn create_terrain_data_with_water_level(&mut self) -> (Vec<Vertex>, Vec<Vertex>) {
        let (cdata, cdata2, ta) = self.terrian_colormap_data();

        let noise_map = self.create_noise_map(self.width, self.height);
       
        let mut data:Vec<Vertex> = vec![];
        let mut data2:Vec<Vertex> = vec![];

        for x in 0..self.width as usize {
            for z in 0..self.height as usize {
                let mut y = if noise_map[x][z].is_finite() { noise_map[x][z] } else { 0.0 };
                if y < self.water_level {
                    y = self.water_level - 0.01;
                }

                let position = [x as f32, y, z as f32];
                let color = self.add_terrain_colors(&cdata, &ta, 0.0, 1.0, y);
                let color2 = self.add_terrain_colors(&cdata2, &ta, 0.0, 1.0, y);

                data.push(Vertex { position, color });
                data2.push(Vertex { position, color: color2 });
            }
        }
        (data, data2)   
    }

    pub fn create_terrain_data_chunk(&mut self) -> (Vec<Vertex>, Vec<Vertex>, u32){
        let increment_count = if self.level_of_detail <= 5 { self.level_of_detail + 1} else { 2*(self.level_of_detail - 2)};
        
        let vertices_per_row = (self.chunk_size - 1)/increment_count + 1;

        let (cdata, cdata2, ta) = self.terrian_colormap_data();

        let noise_map = self.create_noise_map(self.chunk_size, self.chunk_size);
       
        let mut data:Vec<Vertex> = vec![];
        let mut data2:Vec<Vertex> = vec![];

        for x in (0..self.chunk_size as usize).step_by(increment_count as usize) {
            for z in (0..self.chunk_size as usize).step_by(increment_count as usize) {
                let mut y = if noise_map[x][z].is_finite() { noise_map[x][z] } else { 0.0 };
                if y < self.water_level {
                    y = self.water_level - 0.01;
                }

                let position = [x as f32, y, z as f32];
                let color = self.add_terrain_colors(&cdata, &ta, 0.0, 1.0, y);
                let color2 = self.add_terrain_colors(&cdata2, &ta, 0.0, 1.0, y);

                data.push(Vertex { position, color });
                data2.push(Vertex { position, color: color2 });
            }
        }
        (data, data2, vertices_per_row)
    }

    
    pub fn create_terrain_data_multiple_chunks(&mut self, x_chunks:u32, z_chunks:u32, translations:&Vec<[f32;2]>)
    -> (Vec<Vec<Vertex>>, Vec<Vec<Vertex>>, u32) {
        let mut data:Vec<Vec<Vertex>> = vec![];
        let mut data2:Vec<Vec<Vertex>> = vec![];
        let mut vertices_per_row = 0u32;

        let mut k:u32 = 0;
        for _i in 0..x_chunks {
            for _j in 0..z_chunks {
                self.offsets = translations[k as usize];
                let dd = self.create_terrain_data_chunk();
                data.push(dd.0);
                data2.push(dd.1);
                vertices_per_row = dd.2;
                k += 1;
            }
        }
        (data, data2, vertices_per_row)
    }
}