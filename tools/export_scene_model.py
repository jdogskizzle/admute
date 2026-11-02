"""Export the image model the scene check uses (CLIP ViT-B/32 image encoder) to ONNX.

Run at image build time (see Dockerfile), so the running containers only need onnxruntime.
    python export_scene_model.py OUT.onnx
"""
import sys

import open_clip
import torch


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual = model.visual

    def forward(self, x):  # x: N x 3 x 224 x 224, already normalised
        f = self.visual(x)
        return f / f.norm(dim=-1, keepdim=True)


model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(Encoder(model), dummy, sys.argv[1], input_names=["image"], output_names=["features"],
                  dynamic_axes={"image": {0: "n"}, "features": {0: "n"}}, opset_version=17, dynamo=False)
print("exported", sys.argv[1])
