# a script to download the model files from huggingface

from huggingface_hub import hf_hub_download
import shutil

downloaded_path = hf_hub_download(repo_id="", filename="yolo_model_complete.h5")

shutil.move(downloaded_path, "")

# NOTE TO SELF: you can use this to load the model_data into keras compatible .h5 if it doesnt work
# def get_classes(classes_path):
#     with open(classes_path) as f:
#         class_names = f.readlines()
#     class_names = [c.strip() for c in class_names]
#     return class_names

# def get_anchors(anchors_path):
#     with open(anchors_path) as f:
#         anchors = f.readline()
#     anchors = [float(x) for x in anchors.split(',')]
#     return np.array(anchors).reshape(-1, 2)


# class_names = get_classes('model_data/coco_classes.txt')
# anchors = get_anchors('model_data/yolo_anchors.txt')

# image_input = Input(shape=(608, 608, 3))
# keras_model = yolo_body(image_input, len(anchors), len(class_names))

# tf_model = tf.saved_model.load('model_data')
# tf_weights = [v.numpy() for v in tf_model.variables]
# keras_model.set_weights(tf_weights)

# keras_model.save('yolo_model_complete.h5')
# keras_model.summary()