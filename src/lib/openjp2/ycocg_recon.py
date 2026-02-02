#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from PIL import Image
import sys

def read_pgx_16(path: Path) -> tuple[np.ndarray, str]:
    data = path.read_bytes()
    nl = data.find(b"\n")
    if nl < 0:
        raise ValueError(f"{path}: no PGX header")

    header = data[:nl].decode("ascii").strip()
    parts = header.split()
    if parts[0] != "PG":
        raise ValueError(f"{path}: not PGX")

    endian = parts[1]   # ML / LM
    sign   = parts[2]   # + / -
    depth  = int(parts[3])
    w      = int(parts[4])
    h      = int(parts[5])

    if depth != 16:
        raise ValueError("expected 16-bit PGX")

    be = (endian == "ML")
    if sign == "+":
        dt = np.dtype(">u2") if be else np.dtype("<u2")
    else:
        dt = np.dtype(">i2") if be else np.dtype("<i2")

    payload = data[nl+1:]
    arr = np.frombuffer(payload, dtype=dt).reshape((h, w)).astype(np.int32)
    return arr, sign

def inverse_rct(Y, Co, Cg):
    # Y: level-shifted
    # Co/Cg: signed residuals
    t = Y - (Cg >> 1)
    G = Cg + t
    B = t - (Co >> 1)
    R = Co + B
    return R, G, B

def window_from_Y(Y0, lo_p=1.0, hi_p=99.0):
    lo = float(np.percentile(Y0, lo_p))
    hi = float(np.percentile(Y0, hi_p))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi

def apply_window(x, lo, hi):
    y = (x.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(y, 0, 255).astype(np.uint8)

def main(y_pgx, co_pgx, cg_pgx, out_png):
    Y,  sY  = read_pgx_16(Path(y_pgx))
    Co, sCo = read_pgx_16(Path(co_pgx))
    Cg, sCg = read_pgx_16(Path(cg_pgx))

    print(f"Y  sign={sY}  min={Y.min()}  max={Y.max()}")
    print(f"Co sign={sCo} min={Co.min()} max={Co.max()}")
    print(f"Cg sign={sCg} min={Cg.min()} max={Cg.max()}")

    # Y must be level-shifted
    if sY != "+":
        Y = Y + 32768

    # Co/Cg must NOT be shifted
    R, G, B = inverse_rct(Y, Co, Cg)

    # undo level shift once, for display
    R0 = R - 32768
    G0 = G - 32768
    B0 = B - 32768
    Y0 = Y - 32768

    lo, hi = window_from_Y(Y0)

    rgb = np.dstack([
        apply_window(R0, lo, hi),
        apply_window(G0, lo, hi),
        apply_window(B0, lo, hi),
    ])

    Image.fromarray(rgb).save(out_png)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: ycocg_recon.py Y.pgx Co.pgx Cg.pgx out.png", file=sys.stderr)
        sys.exit(2)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
