import numpy as np, torch, torch.nn as nn
from PIL import Image
from denoising.models import FFDNet

SIGMA = 25 / 255  # FFDNet noise level (repo default)

ff = FFDNet(num_input_channels=3)
sd = torch.load("denoising/models/net_rgb.pth", map_location="cpu")
sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
ff.load_state_dict(sd)
ff.eval()

# ONNX-friendly FFDNet: PixelUnshuffle/Shuffle match the repo's manual
# space-to-depth channel order (channel = c*4 + row*2 + col, noise map first).
class FFDNetOnnx(nn.Module):
    def __init__(self, ff, sigma):
        super().__init__()
        self.dncnn = ff.intermediate_dncnn
        self.us = nn.PixelUnshuffle(2)
        self.sh = nn.PixelShuffle(2)
        self.sigma = sigma
    def forward(self, x):                      # x: [1,3,H,W], H/W even, [0,1]
        down = self.us(x)                      # [1,12,H/2,W/2]
        n, _, h, w = down.shape
        noise = torch.full((n, 3, h, w), self.sigma, dtype=x.dtype)
        pred = self.dncnn(torch.cat([noise, down], 1))   # [1,12,...]
        pred = self.sh(pred)                   # [1,3,H,W] predicted noise
        return torch.clamp(x - pred, 0.0, 1.0)

wrap = FFDNetOnnx(ff, SIGMA).eval()

# 1) validate the wrapper vs the repo's own FFDNet forward (proves channel order)
x = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    ref = torch.clamp(x - ff(x, torch.FloatTensor([SIGMA])), 0.0, 1.0)
    mine = wrap(x)
print("wrapper-vs-repo maxdiff:", (mine - ref).abs().max().item())

# 2) export
dummy = torch.rand(1, 3, 256, 256)
torch.onnx.export(
    wrap, dummy, "denoiser.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}},
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print("WROTE denoiser.onnx")

# 3) validate ONNX vs torch on a real page (even dims)
import onnxruntime as ort
im = Image.open("figures/bw3.jpg").convert("RGB")
w, h = im.size; w -= w % 2; h -= h % 2
im = im.resize((w, h))
arr = (np.asarray(im, np.float32) / 255).transpose(2, 0, 1)[None].copy()
with torch.no_grad():
    t_out = wrap(torch.from_numpy(arr)).numpy()
sess = ort.InferenceSession("denoiser.onnx", providers=["CPUExecutionProvider"])
o_out = sess.run(["output"], {"input": arr})[0]
print("onnx-vs-torch maxdiff:", float(np.abs(t_out - o_out).max()))
Image.fromarray((np.clip(o_out[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)).save("denoised.png")
print("saved denoised.png")
