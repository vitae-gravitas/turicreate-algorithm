import os
from utils import Utils
from validateData import cleanData
from dataaug import augmentData
import config as cfg
import xmlToSFrame as converter

def run(sframe_dir='sframe', config=True):
	"""
	Runs the full stack starting with ImageNet data to the finalized model.

	This function will ask the user to input a playground directory, annotations directory, image directory, and a testing images directory. It will generate several folders in the playground directory containing intermediaries of the results: pickled versions of annotation data, the SFrame, and the finalized model.

	Parameters
	----------
	pickle_dir: str, optional
		The name of the auto-generated folder that contains the pickle files.
	sframe_dir: str
		The name of the auto-generated folder that contains the SFrame file.
	model_dir: float, optional
		The name of the auto-generated folder that contains the model files.
	config: bool, optional
		If true, then will grab directory data from a config file. Otherwise, will prompt the user during runtime.
	"""

	if config:
		playground_dir = cfg.playground_dir 
		annotatations_dir = cfg.annotatations_dir 
		image_dir = cfg.image_dir if config else None
	else:
		playground_dir = input("Playground Directory: ").strip()
		annotatations_dir = input("Annotations Directory (this can be unclean): ").strip()
		image_dir = input("Images Directory (this can be unclean): ").strip()

	#Clean the Data
	allImages_dir = os.path.join(playground_dir, "allImages")
	allImages_created = Utils.make_dir(allImages_dir)
	if allImages_created:
		print("\nCLEANING DATA...\n")
		cleanData(annotatations_dir, image_dir, allImages_dir)
	else:
		print("\nUSING CLEANED DATA FROM PREV RUN...\n")

	sframe_dir = os.path.join(playground_dir, sframe_dir)
	sframe_created = Utils.make_dir(sframe_dir)

	if sframe_created:
		print("\nCONVERTING XML ANNOTATIONS TO SFRAME...\n")
		converter.createSFrame(annotatations_dir, allImages_dir, sframe_dir)
		print()
	else:
		print("\nSFRAME FOUND...\n")

	# Augmenting the data
	allAugmentedImages_dir = os.path.join(playground_dir, "allAugmentedImages")
	allAugmentedImages_created = Utils.make_dir(allAugmentedImages_dir)
	if allAugmentedImages_created:
		print("\nGENERATING AUGMENTATIONS\n")
		augmentData(sframe_dir, allAugmentedImages_dir)
		print()
	else:
		print("\nUSING AUGMENTED DATA FROM PREV RUN...\n")

	#explore the dataset if requested
	if True:
		explore_req = raw_input("Explore a sFrame? (y/n)")
		if explore_req == 'y':
			converter.explore(config.training_sFrame)

	# print("\nLET'S BEGIN TRAINING...\n")
	# #Begin training
	# training_sFrame = cfg.training_sFrame if CONFIG else input("Which sFrame directory would you like to train on?: ").strip()
	# model_dir = os.path.join(playground_dir, model_dir)
	# Utils.make_dir(model_dir) #Will always write into the same folder, even if it exists...

	# train_test_split = cfg.train_test_split if CONFIG else float(input("Train test split (from 0-1): ").strip())
	# max_iterations = cfg.max_iterations if CONFIG else int(input("Maximum Iterations: ").strip())
	# model_name = cfg.model_name if CONFIG else input("Model Name: ").strip()
	# # test_images_dir = input("Test Images Directory: ").strip()

	# print("\nTRAINING...\n")
	# mh.train(training_sFrame, model_dir, train_test_split, max_iterations, model_name)
	# mh.visualize_results(model_dir, test_images_dir, model_name)

