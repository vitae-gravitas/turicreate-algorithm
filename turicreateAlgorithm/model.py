import turicreate as tc
import os as os

from tc_runner.utils import Utils

class SFrameException(Exception):
	pass

class ModelHandler(object):
	def __init__(self):
		tc.config.set_num_gpus(0)
		pass

	@staticmethod
	def train(sframe_dir, model_dir, train_test_split=0.8, max_iterations=5000, model_name='mymodel'):
		"""
		Trains a model using the SFrame.

		Parameters
		----------
		sframe_dir: str
			The folder path containing the SFrame.
		model_dir: str
			The folder path to write the model files in.
		train_test_split: float, optional
			The ratio of training data to testing data.			
		max_iterations: int, optional
			Number of iterations to train the model.
		model_name: str, optional
			The name of the model file (to be written).
		"""
		# for file in os.listdir(sframe_dir):
		# 	if file.endswith('.sframe'):
		# 		data = tc.SFrame(os.path.join(sframe_dir, file))

		data = tc.SFrame(os.path.join(sframe_dir))

		train_data, test_data = data.random_split(train_test_split)
		model = tc.object_detector.create(train_data, feature='image', annotations='annotations', max_iterations=max_iterations)
		# Save predictions to an SArray
		predictions = model.predict(test_data)

		# Evaluate the model and save the results into a dictionary
		metrics = model.evaluate(test_data)

		#Save performance metrics
		Utils.write_as_pckl(metrics, "metrics", model_dir)

		# Save the model for later use in Turi Create
		model.save(os.path.join(model_dir, '{0}.model'.format(model_name)))

		# Export for use in Core ML
		model.export_coreml(os.path.join(model_dir, '{0}.mlmodel'.format(model_name)))

	@staticmethod
	def visualize_results(model_dir, test_images_dir, model_name='mymodel'):
		"""
		Tests the model and visualizes the predicted bounding boxes.

		Parameters
		----------
		model_dir: str
			The folder path containing the model files.
		test_images_dir: str
			The folder path containing the testing images.
		model_name: str, optional
			The filename of the written model.
		"""
		model = tc.load_model(os.path.join(model_dir, '{0}.model'.format(model_name)))

		sf_images = tc.image_analysis.load_images(test_images_dir, recursive=True, random_order=True)
		sf_test = tc.SFrame({'image': [tc.Image(path) for path in sf_images['path']]})
		sf_test['predictions'] = model.predict(test, confidence_threshold=0.2)

		sf_test['image_with_predictions'] = tc.object_detector.util.draw_bounding_boxes(sf_test['image'], sf_test['predictions'])
		sf_test[['image', 'predictions', 'image_with_predictions']].explore()


