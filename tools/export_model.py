from pathlib import Path

import torch
from efficientnet_pytorch import EfficientNet


model_path = "/datadrive/rlan/models/bestmodel_2.pth"
onnx_save_path = "/tmp/models/ccth_subbrand_classification/1/model.onnx"
chkp = torch.load(model_path)
model = EfficientNet.from_name("efficientnet-b5", override_params={'num_classes': 197})
model.load_state_dict(chkp["model"])

# model = EfficientNet.from_pretrained("efficientnet-b0")
model.set_swish(memory_efficient=False)
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    onnx_save_path,
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable lenght axes
    verbose=True,
)

# import onnx
# model = onnx.load("/datadrive/rlan/models/bestmodel_2.onnx")
# # Check that the IR is well formed
# onnx.checker.check_model(model)
#
# # Print a human readable representation of the graph
# onnx.helper.printable_graph(model.graph)

# import onnxruntime
# x = torch.randn(2, 3, 256, 256, requires_grad=True)  # you can change batch size 10 to other numbers
#
# ort_session = onnxruntime.InferenceSession(onnx_save_path)
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
#
# ort_outs = ort_session.run(None, ort_inputs)
#
# print(ort_outs)
# print(ort_outs[0].shape)

