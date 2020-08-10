import cv2
import numpy as np
import onnxruntime


def run_with_onnx_runtime(model_path, test_image_path, w=225, h=225):
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name

    image = cv2.imread(test_image_path)
    image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    x = image_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    x = np.ascontiguousarray(x)
    x = x.astype(np.float32)
    x /= 255.0
    if image.ndim == 3:
        x = np.expand_dims(x, 0)
    output = session.run([], {input_name: x})
    print(output)


if __name__ == '__main__':
    run_with_onnx_runtime('yolov3.onnx', 'demo/test1.jpg', w=608, h=608)
