"""Test the trained model."""
from test import Test

if (__name__ == '__main__'):
    image_path = 'test.jpg'
    model_path = 'model/VGG16_modelParams.ckpt'
    test = Test(image=image_path, graph_path=model_path)
    flower_category = test.run_graph()
