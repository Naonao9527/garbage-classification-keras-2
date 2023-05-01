from tensorflow import lite
import config,sys
sys.path.append(r"E:\文档\目标检测工程集\garbage-classification-keras/")
from nets import yolo

confidence      = 0.5
input_shape     = [416, 416]
anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

def generate():
    assert config.model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
    
    model = yolo.yolo_body([input_shape[0], input_shape[1], 3], anchors_mask, config.num_classes)
    model.load_weights(config.model_path,by_name=True, skip_mismatch=True)

    print('{} model, anchors, and classes loaded.'.format(config.model_path))

    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
    lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    lite.Optimize.OPTIMIZE_FOR_SIZE,
    ]
    # converter.target_spec.supported_types = [lite.constants.FLOAT16]
    tflite_model = converter.convert()

    # Save the model.
    with open('raspi/model.tflite', 'wb') as f:
      f.write(tflite_model)
  
if __name__ =="__main__":
    generate()
