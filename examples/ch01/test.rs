use noise::{ NoiseFn, Perlin };

fn main() {
    let perlin = Perlin::new(1232);
    let noise_value = perlin.get([x, y]); // Replace x, y with your coordinates
    println!("Perlin Noise Value: {}", noise_value);
}
