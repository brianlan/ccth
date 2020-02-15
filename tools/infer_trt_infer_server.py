import numpy as np
import requests


image = np.random.rand(1,6,512,512)
image = image.astype('float32')
byte_batch=[]
byte_batch.append(image.tobytes())
trt_ip='127.0.0.1:8000'
model_name='onnx'
model_version='1'
Header = {
  'NV-InferRequest': " ".join(str(irh).split()),
  'Content-Type': 'application/octet-stream'
}
r = requests.post(url='http://{}/api/infer/{}/{}?format=binary'.format(trt_ip, model_name, model_version),
                  data=b''.join(byte_batch), headers=Header)