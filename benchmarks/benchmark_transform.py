import time
from io import BytesIO
from PIL import Image
from zendolib import transform_image

IMAGE_PATH = "../backend/python/app/uploads/6e0fbd28-e451-4dc8-90b5-6fc25b25c106.webp"
SCALE = 0.5
N = 10

def benchmark_pil():
    start = time.perf_counter()
    for _ in range(N):
        with Image.open(IMAGE_PATH) as img:
            w, h, = img.size
            img.resize((int(w * SCALE), int(h * SCALE)), resample=Image.Resampling.LANCZOS)
            raw = img.convert("RGB").tobytes()

        end = time.perf_counter()
        return (end - start) * 1000 / N

def benchmark_rust():
    start = time.perf_counter()
    for _ in range(N):
        raw = transform_image(IMAGE_PATH, SCALE)
    end = time.perf_counter()
    return (end - start) * 1000 / N


if __name__ == "__main__":
    pil_time = benchmark_pil()
    rust_time = benchmark_rust()

    print(f"PIL resize avg time: {pil_time:.2f} ms")
    print(f"Rust resize avg time: {rust_time:.2f} ms")
    if pil_time:
        print(f"Rust is {pil_time / rust_time:.2f}x faster" if rust_time < pil_time else f"PIL is {rust_time / pil_time:.2f}x faster")
