#!/usr/bin/env python3
"""Plaintext reference forward of the fused ResNet-20 (resnet20_fused/ npy).

Prints the true label and 10 logits for test image(s) — the FHE run's logits
must match these up to relu-approximation + CKKS noise (~0.1-0.3 abs).

Usage (from the repo root):
  python3 unittest/resnet/plain_forward.py [num_images]
"""
import sys

import numpy as np

WEIGHT_DIR = "resnet20_fused"
CIFAR_BIN = "cifar10_data/cifar-10-batches-bin/test_batch.bin"
MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
STD = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)


def load(name):
    return np.load(f"{WEIGHT_DIR}/{name}")


def conv2d(x, w, b, stride=1):
    """x: (C_in,H,W), w: (C_out,C_in,3,3) or (C_out,C_in,1,1), pad=k//2."""
    c_out, c_in, k, _ = w.shape
    pad = k // 2
    h, wd = x.shape[1], x.shape[2]
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh, ow = h // stride, wd // stride
    out = np.zeros((c_out, oh, ow))
    for oy in range(oh):
        for ox in range(ow):
            patch = xp[:, oy * stride:oy * stride + k, ox * stride:ox * stride + k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2]))
    return out + b.reshape(-1, 1, 1)


def relu(x):
    return np.maximum(x, 0)


def block(x, prefix, narrowing):
    w1 = load(f"{prefix}.conv1_reparam.weight")
    b1 = load(f"{prefix}.conv1_reparam.bias")
    w2 = load(f"{prefix}.conv2_reparam.weight")
    b2 = load(f"{prefix}.conv2_reparam.bias")
    stride = 2 if narrowing else 1
    t = relu(conv2d(x, w1, b1, stride))
    t = conv2d(t, w2, b2, 1)
    if narrowing:
        ws = load(f"{prefix}.shortcut_reparam.weight")
        bs = load(f"{prefix}.shortcut_reparam.bias")
        if ws.ndim == 2:
            ws = ws.reshape(ws.shape[0], ws.shape[1], 1, 1)
        ident = conv2d(x, ws, bs, 2)
    else:
        ident = x
    return relu(t + ident)


DEBUG = False


def dbg(name, x):
    # FHE ciphertexts carry x/10 (relu_range normalization); print in that unit
    if DEBUG:
        n = x / 10.0
        print(f"[plain] {name} max|x|/10={np.abs(n).max():.6f} "
              "first: " + " ".join(f"{v:.6f}" for v in n[0, 0, :6]))


def forward(img):
    x = conv2d(img, load("conv1_reparam.weight"),
               load("conv1_reparam.bias"), 1)
    dbg("conv0", x)
    x = relu(x)
    dbg("relu0", x)
    for i in range(3):
        x = block(x, f"layer1.{i}", False)
        dbg(f"block1_{i+1}", x)
    x = block(x, "layer2.0", True)
    dbg("block2_1", x)
    for i in (1, 2):
        x = block(x, f"layer2.{i}", False)
        dbg(f"block2_{i+1}", x)
    x = block(x, "layer3.0", True)
    dbg("block3_1", x)
    for i in (1, 2):
        x = block(x, f"layer3.{i}", False)
        dbg(f"block3_{i+1}", x)
    feat = x.mean(axis=(1, 2))                      # global avg pool -> (64,)
    fc_w = load("linear.weight")
    fc_b = load("linear.bias")
    return fc_w @ feat + fc_b


def main():
    global DEBUG
    if "--debug" in sys.argv:
        DEBUG = True
        sys.argv.remove("--debug")
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    raw = open(CIFAR_BIN, "rb").read()
    for i in range(num):
        rec = raw[i * 3073:(i + 1) * 3073]
        label = rec[0]
        img = np.frombuffer(rec[1:], dtype=np.uint8).astype(np.float64)
        img = img.reshape(3, 32, 32) / 255.0
        img = (img - MEAN) / STD
        logits = forward(img)
        print(f"img {i} true label {label} logits:",
              " ".join(f"{v:.5f}" for v in logits),
              f"-> pred {int(np.argmax(logits))}")


if __name__ == "__main__":
    main()
