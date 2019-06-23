import os

dir_path = os.path.dirname(os.path.realpath(__file__))

playground_dir = os.path.join(dir_path, "Playground")
annotatations_dir = os.path.join(dir_path, "annotations")
image_dir = os.path.join(dir_path, "images")
training_sFrame = os.path.join(dir_path, "Playground/sframe/final.sframe")
train_test_split = 0.8
max_iterations = 10000
model_name = "heavy_aug"
