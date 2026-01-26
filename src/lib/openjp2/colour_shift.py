import numpy as np

def read_pgx(path: str) -> np.ndarray:
    # PGX format: first line "PG ML +|-" then "width height" then "maxval" then raw samples (big-endian)
    with open(path, "rb") as f:
        header1 = f.readline().decode("ascii").strip()
        header2 = f.readline().decode("ascii").strip()
        header3 = f.readline().decode("ascii").strip()

        parts = header1.split()
        if len(parts) < 4 or parts[0] != "PG":
            raise ValueError(f"{path}: not a PGX file? header1={header1!r}")

        endian = parts[1]     # "ML" (big-endian) or "LM" (little-endian)
        signed = parts[2]     # "+" unsigned, "-" signed
        bitdepth = int(parts[3])

        w, h = map(int, header2.split())
        maxval = int(header3)  # usually (1<<bitdepth)-1, but we don't strictly need it

        if bitdepth <= 8:
            dtype = np.int8 if signed == "-" else np.uint8
        elif bitdepth <= 16:
            dtype = np.int16 if signed == "-" else np.uint16
        else:
            dtype = np.int32 if signed == "-" else np.uint32

        # PGX samples are stored as 2-byte words for <=16, 4-byte for >16, etc.
        # NumPy dtype already encodes element size.
        raw = f.read()

        dt = np.dtype(dtype)
        if (endian == "ML"):  # big-endian
            dt = dt.newbyteorder(">")
        else:                 # "LM" little-endian
            dt = dt.newbyteorder("<")

        arr = np.frombuffer(raw, dtype=dt)
        if arr.size != w * h:
            raise ValueError(f"{path}: expected {w*h} samples, got {arr.size}")
        return arr.reshape((h, w))

def stats(name: str, a: np.ndarray):
    a64 = a.astype(np.int64)
    print(f"{name}: dtype={a.dtype} shape={a.shape} min={a64.min()} max={a64.max()} mean={a64.mean():.2f}")

Y  = read_pgx("/tmp/out_0.pgx")
Co = read_pgx("/tmp/out_1.pgx")
Cg = read_pgx("/tmp/out_2.pgx")

stats("Y", Y)
stats("Co", Co)
stats("Cg", Cg)
