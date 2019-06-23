import os
import shutil

class DataCleanerHandler(object):
	def __init__(self, annotatations_dir=None, image_dir=None):
		"""
		Sets up the handler by getting the directories and counting the associated files in each one.

		Parameters
		----------
		annotatations_dir : string, optional
			Directory to annotations folder.

		image_dir : string, optional
			 Directory to image folder.
		"""
		if not annotatations_dir or not image_dir:
			annotatations_dir = input("Annotations Directory: ").strip()
			image_dir = input("Images Directory: ").strip()
		self.annotatations_dir = annotatations_dir
		self.image_dir = image_dir

		self.annotationFileNames = self.getFileSet(annotatations_dir, ".xml")
		self.imageFileNames = self.getFileSet(image_dir, ".JPEG")

		print("Cleaning data for " + os.path.basename(self.annotatations_dir))

		print("\tAnnotation Files Count: " + str(len(self.annotationFileNames)))
		print("\tImage Files Count: " + str(len(self.imageFileNames)))

	def getFileSet(self, dir, file_ext):
		fileSet = set();
		for file in os.listdir(dir):
			if file.endswith(file_ext):
				fileSet.add(file.split(".")[0]);
		return fileSet

	def isDataValid(self):
		self.newAnnotationFileNames = self.getFileSet(self.annotatations_dir, ".xml")
		self.newImageFileNames = self.getFileSet(self.image_dir, ".JPEG")

		if (self.newAnnotationFileNames.issubset(self.newImageFileNames) and self.newImageFileNames.issubset(
				self.newAnnotationFileNames)):
			print("\tData is valid!")
			return True
		else:

			raise Exception("\tData is still not valid, something went wrong!")

	def deleteFiles(self):
		# Delete Images

		commonFiles = self.annotationFileNames.intersection(self.imageFileNames)
		imageFilesToDelete = self.imageFileNames.difference(commonFiles)
		annotationFilesToDelete = self.annotationFileNames.difference(commonFiles)

		print("\tnumber of common files: " + str(len(commonFiles)))

		# Delete ImageFiles

		for delFile in annotationFilesToDelete:
			os.remove(self.annotatations_dir + "/" + delFile + ".xml")

		# Delete AnnotationFiles

		for delFile in imageFilesToDelete:
			os.remove(self.image_dir + "/" + delFile + ".JPEG")

def cleanData(annotatations_dir, image_dir, allImages_dir):
	"""
	This method will call upon the methods of DataCleanerHandler in order to clean the data downloaded from ImageNet.
	It will read through the annoation folder and image folder and find a common set of file names that are shared. It
	will then delete all the files that are not share in both the image and annotation folders.

	Parameters
	----------
	annotatations_dir : str, optional
		The directory where the annotations are stored.
	image_dir : str, optional
		The directory where the images are stored.
	"""

	for obj_label in os.listdir(image_dir):
		img_path = os.path.join(image_dir, obj_label)
		ann_path = os.path.join(annotatations_dir, obj_label)
		if os.path.isdir(img_path) and os.path.isdir(ann_path):
			cleaner = DataCleanerHandler(ann_path, img_path)
			cleaner.deleteFiles()
			cleaner.isDataValid()
			recursive_copy(img_path, allImages_dir)
		else:
			pass

def recursive_copy(src, dest):
	"""
	Copy each file from src dir to dest dir, including sub-directories.
	"""
	for item in os.listdir(src):
		file_path = os.path.join(src, item)

		# if item is a file, copy it
		if os.path.isfile(file_path):
			shutil.copy(file_path, dest)

		# else if item is a folder, recurse
		elif os.path.isdir(file_path):
			new_dest = os.path.join(dest, item)
			os.mkdir(new_dest)
			recursive_copy(file_path, new_dest)