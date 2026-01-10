# recover_tensors.py
import sys, struct, json, pickle
import numpy as np
import torch
from safetensors.torch import save_file

def try_pickle(b):
    try:
        return pickle.loads(b)
    except Exception:
        return None

DTYPE_MAP = {
    "F16": np.float16, "F32": np.float32, "F64": np.float64,
    "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64,
    "BF16": np.uint16,  # BF16 stored as uint16 in safetensors
}

def main(path):
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 8:
        print("File too small to be safetensors.")
        return

    # Try to parse safetensors header (8-byte LE length + JSON header + padding to 8)
    header_len = struct.unpack("<Q", buf[:8])[0]
    header_bytes = buf[8:8+header_len]
    data_start = 8 + header_len
    pad = (8 - (data_start % 8)) % 8
    data_start += pad

    recovered = {}

    # First, see if header is valid JSON
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as e:
        print("Header not JSON, trying whole-file pickle recovery…", e)
        obj = try_pickle(buf)
        if isinstance(obj, dict):
            # maybe you wrote a pickled dict straight to disk
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    recovered[k] = v.detach().cpu().to(torch.float16).contiguous()
                elif isinstance(v, np.ndarray):
                    recovered[k] = torch.from_numpy(v).to(torch.float16).contiguous()
                else:
                    try:
                        recovered[k] = torch.tensor(v, dtype=torch.float16)
                    except Exception:
                        pass
        else:
            print("Whole-file pickle failed; no recovery.")
    else:
        # Header parsed: in valid safetensors it's a dict of {name: {dtype, shape, data_offsets}}
        # Your malformed file might still have these; we'll try.
        print("Header parsed. Keys:", list(header.keys()))

        # Some writers include "__metadata__" at top—ignore it if present
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(info, dict):
                continue

            # If it's our *wrong* format (you embedded 'data' inside), try to unpickle it
            if "data" in info and isinstance(info["data"], list):
                # 'data' shouldn’t be here in real safetensors headers, but try to join bytes
                try:
                    raw = bytes(info["data"])
                    obj = try_pickle(raw)
                    if isinstance(obj, torch.Tensor):
                        recovered[name] = obj.detach().cpu().to(torch.float16).contiguous()
                        continue
                except Exception:
                    pass

            # Normal safetensors path
            if all(k in info for k in ("dtype", "shape", "data_offsets")):
                start, end = info["data_offsets"]
                chunk = buf[data_start + start : data_start + end]
                # Try pickle first (in case your "tensor" was actually a pickled blob)
                obj = try_pickle(chunk)
                if isinstance(obj, torch.Tensor):
                    recovered[name] = obj.detach().cpu().to(torch.float16).contiguous()
                    continue
                # Else reconstruct raw array
                dtype = DTYPE_MAP.get(info["dtype"])
                shape = tuple(info["shape"]) if isinstance(info["shape"], list) else None
                if dtype is None or shape is None:
                    print(f"Skipping {name}: unknown dtype/shape {info.get('dtype')}, {info.get('shape')}")
                    continue
                try:
                    arr = np.frombuffer(chunk, dtype=dtype)
                    if np.prod(shape) != arr.size:
                        print(f"Size mismatch for {name}: header {shape} vs bytes {arr.size}")
                        continue
                    arr = arr.reshape(shape)
                    ten = torch.from_numpy(arr)
                    # If BF16 case, reinterpret to torch.bfloat16
                    if info["dtype"] == "BF16":
                        ten = ten.view(torch.bfloat16)
                    recovered[name] = ten.to(torch.float16).contiguous()
                except Exception as e:
                    print(f"Failed to rebuild {name}:", e)

    if not recovered:
        print("No tensors recovered. :(")
        return

    print("Recovered tensors:", {k: tuple(v.shape) for k,v in recovered.items()})
    # Normalize to expected names if possible
    out = {}
    if "embedding_1" in recovered and "embedding_2" in recovered:
        out = {"embedding_1": recovered["embedding_1"], "embedding_2": recovered["embedding_2"]}
    else:
        # pick two largest 1D tensors as best guess
        one_d = [(k, v) for k, v in recovered.items() if v.ndim == 1]
        one_d.sort(key=lambda kv: kv[1].numel(), reverse=True)
        for i, (k, v) in enumerate(one_d[:2], 1):
            out[f"embedding_{i}"] = v

    save_file(out, "recovered.safetensors")
    print("Wrote recovered.safetensors")
    # Quick load-test
    from safetensors.torch import load_file
    t = load_file("recovered.safetensors")
    print("Load test OK. Keys:", list(t.keys()))
    for k in t.keys():
        print(k, t[k].dtype, t[k].shape)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python recover.py <path_to_bad_file.safetensors>")
        sys.exit(1)
    main(sys.argv[1])
