@group(0) @binding(0) var<storage, read_write> indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> indices2: array<u32>;
@group(0) @binding(2) var<uniform> resolution: u32;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(workgroup_id) wid : vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let i = lid.x + wid.x * 8u;
    let j = lid.y + wid.y * 8u;  

    if(i >= resolution - 1u || j >= resolution - 1u ) { return; } 

    let idx = (i + j * (resolution - 1u)) * 6u;

    // first triangle
    indices[idx] = i + j * resolution;
    indices[idx + 1u] = i + (j + 1u) * resolution;
    indices[idx + 2u] = i + 1u + j * resolution;

    // second triangle
    indices[idx + 3u] = i + 1u + j * resolution;
    indices[idx + 4u] = i + (j + 1u) * resolution;
    indices[idx + 5u] = i + 1u + (j  + 1u) * resolution;

    // wireframe
    let tdx = (i + j * (resolution - 1u)) * 8u;
    indices2[tdx] = i + j * resolution;
    indices2[tdx + 1u] = i + 1u + j * resolution;
    indices2[tdx + 2u] = i + 1u + j * resolution;
    indices2[tdx + 3u] = i + 1u + (j + 1u) * resolution;
    indices2[tdx + 4u] = i + 1u + (j + 1u) * resolution;
    indices2[tdx + 5u] = i + (j + 1u) * resolution;
    indices2[tdx + 6u] = i + (j + 1u) * resolution;
    indices2[tdx + 7u] = i + j * resolution;    
}