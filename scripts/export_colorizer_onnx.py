import numpy as np, torch, torch.nn as nn
from PIL import Image
from networks.models import Generator

class Wrap(nn.Module):
    def __init__(self, gen):
        super().__init__(); self.gen = gen
    def forward(self, x):
        return self.gen(x)[0]

gen = Generator()
gen.load_state_dict(torch.load("networks/generator.zip", map_location="cpu"), strict=False)
gen.eval()
model = Wrap(gen).eval()

# ---- export: legacy TorchScript exporter -> single self-contained file ----
dummy = torch.zeros(1, 5, 576, 576); dummy[:, 0] = 0.5
torch.onnx.export(
    model, dummy, "colorizer.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "n", 2: "h", 3: "w"},
                  "output": {0: "n", 2: "h", 3: "w"}},
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print("WROTE colorizer.onnx (legacy)")

# ---- build a real 5ch input from a sample page ----
def preprocess(path, short=576):
    im = Image.open(path).convert("L")           # grayscale
    w, h = im.size
    if h <= w:                                    # landscape: short side = height
        nh = short; nw = max(32, round(w * short / h))
    else:
        nw = short; nh = max(32, round(h * short / w))
    nh += (32 - nh % 32) % 32; nw += (32 - nw % 32) % 32
    im = im.resize((nw, nh), Image.BILINEAR)
    g = np.asarray(im, np.float32) / 255.0        # [0,1], channel 0
    x = np.zeros((1, 5, nh, nw), np.float32)
    x[0, 0] = g                                   # ch1..4 stay 0 (no hint/mask)
    return x

x = preprocess("figures/bw1.jpg")
with torch.no_grad():
    t_out = model(torch.from_numpy(x)).numpy()

import onnxruntime as ort
sess = ort.InferenceSession("colorizer.onnx", providers=["CPUExecutionProvider"])
o_out = sess.run(["output"], {"input": x})[0]

diff = np.abs(t_out - o_out)
print(f"validate: torch{t_out.shape} vs onnx{o_out.shape} maxdiff={diff.max():.3e} meandiff={diff.mean():.3e}")

def save(arr, path):
    img = (np.clip(arr[0].transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
save(t_out, "out_torch.png"); save(o_out, "out_onnx.png")
print("saved out_torch.png / out_onnx.png")
