import numpy as np
# import sys
# print(sys.path)
# import cvml_common.predictor.trt_tools as trt_tools # see trt_tools.py for this part
from collections import defaultdict
# import os
# import sys
import re
# import time
import requests
import cvml_common.predictor.src.core.api_pb2 as api
# import cvml_common.predictor.src.core.grpc_service_pb2 as grpc_service
# import cvml_common.predictor.src.core.model_config_pb2 as model_config
# import cvml_common.predictor.src.core.request_status_pb2 as request_status
# import cvml_common.predictor.src.core.server_status_pb2 as server_status

# input_name, _1, output_names = trt_tools.parse_model('onnx', '127.0.0.1:8000')
#print(input_name, _1, output_names)
image = np.random.rand(3,256,256)
image = image.astype('float32')
byte_batch=[]
byte_batch.append(image.tobytes())
trt_ip='127.0.0.1:8000'
model_name='ccth_subbrand_classification'
model_version='1'
irh = api.InferRequestHeader()

# Add outputs
input = irh.input.add()
input.name = 'input'
input_dims=[3,256,256]
input.dims.extend(input_dims)

# Add outputs
output = irh.output.add()
output.name = "output"

# Add batch_size
irh.batch_size = len(byte_batch)

# Send Request
Header = {
  'NV-InferRequest': " ".join(str(irh).split()),
  'Content-Type': 'application/octet-stream'
}
# print Header
r = requests.post(url='http://{}/api/infer/{}/{}?format=binary'.format(trt_ip, model_name, model_version),
                  data=b''.join(byte_batch), headers=Header)
infer_results = []
if r.status_code == 200:
  # Refer https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/http_grpc_api.html#inference
  # Process outputs
  bbs = re.findall(r'batch_byte_size: (\d*)', r.headers['NV-InferResponse'])
  byte_size = 0
  for bs in bbs:
    byte_size += int(bs)

  response = api.InferResponseHeader()
  response.ParseFromString(r.content[byte_size:])

  results = []
  ptr = 0
  for output in response.output:
    if output.HasField('raw'):
      output_floats = np.frombuffer(r.content[ptr:ptr + output.raw.batch_byte_size], np.float32)
      if len(output.raw.dims) > 1:
        dims = tuple(output.raw.dims)
        output_floats = output_floats.reshape(dims)
      elif len(byte_batch) > 1:
        output_floats = output_floats.reshape((len(byte_batch), output.raw.dims[0]))
      results.append((output.name, output_floats))
      ptr += output.raw.batch_byte_size
    else:
      # If not raw, output cls is requested
      # In that case, classification results are inside response
      output_dicts = []
      for classes in output.batch_classes:
        prediction_dict = defaultdict(float)
        for cls in classes.cls:
          prediction_dict[int(cls.label)] = cls.value
        output_dicts.append(prediction_dict)
      results.append((output.name, output_dicts))
  infer_results.append(results)
print(infer_results)
