use std::path::Path;
use zendolib::transform_image;

fn main() {
    let path = Path::new("../python/app/uploads/6e0fbd28-e451-4dc8-90b5-6fc25b25c106.webp");
    let result = transform_image(path.to_str().unwrap(), 0.5).expect("Failed to transform image");

    result.save("output.jpg").unwrap();
}
