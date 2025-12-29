import tensorflow as tf

interpreter = tf.lite.Interpreter(
    # model_path="embed_model_int8.tflite"
    model_path="/home/kend/文档/PlatformIO/Projects/DogStopper/python/tinyml/train/quantized_model_mfcc_int8.tflite"
)
interpreter.allocate_tensors()

ops = interpreter._get_ops_details()

op_set = sorted({op["op_name"] for op in ops})

print("=== Ops used in model ===")
for op in op_set:
    print(op)
