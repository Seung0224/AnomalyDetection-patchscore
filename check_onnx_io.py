import onnxruntime as rt

sess = rt.InferenceSession("D:/ADI/wrn50_l2l3.onnx", providers=["CPUExecutionProvider"])

print("=== Inputs ===")
for i in sess.get_inputs():
    print(i.name, i.shape, i.type)

print("=== Outputs ===")
for o in sess.get_outputs():
    print(o.name, o.shape, o.type)