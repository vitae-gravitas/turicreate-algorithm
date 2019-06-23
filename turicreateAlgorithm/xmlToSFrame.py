import os
import turicreate as tc

from utils import Utils

try:
	import xml.etree.cElementTree as et
except ImportError:
	import xml.etree.ElementTree as et


class AnnotationError(Exception):
	pass


class ImageHandler(object):
	def __init__(self):
		self.root_dir = os.path.dirname(os.path.realpath(__file__))

	def parse_xml(self, label, xml_tree, image_dir, img_type='JPEG'):
		"""
		Grabs relevant data from the xml_tree.

		Retains data on the label of the annotated object, as well as the minimum and maximum xy coordinates of the bounding box(es).

		Parameters
		----------
		label : str
			The label of the annotated object, for e.g. 'barbell'.
		xml_tree: ElementTree
			Data type representing the data in the xml file.
		image_dir: str
			The folder path containing the annotated image files.
		img_type: str, optional
			The image extension, for e.g. 'JPEG'.

		Returns
		-------
		dict
			Data points of interest stored as key value pairs.
		"""
		data = {}
		bounding = {
			'label': label,
			'xMin': [],
			'xMax': [],
			'yMin': [],
			'yMax': [],
		}

		for elem in xml_tree.iter():
			tag = elem.tag
			attr = elem.text

			if tag == 'filename':
				data['path'] = os.path.join(image_dir, "{0}.{1}".format(attr, img_type))
			elif tag == 'xmin':
				bounding['xMin'].append(int(attr))
			elif tag == 'ymin':
				bounding['yMin'].append(int(attr))
			elif tag == 'xmax':
				bounding['xMax'].append(int(attr))
			elif tag == 'ymax':
				bounding['yMax'].append(int(attr))

		data['annotations'] = ImageHandler.transform_bounding(bounding)

		return data

	@staticmethod
	def transform_bounding(bounding):
		"""
		Mutates the bounding box data to better fit TuriCreate's standards.

		Changes data of min/max coordinates to height, width, center points.

		Parameters
		----------
		bounding : dict
			Key value pairs for the bounding box data.

		Returns
		-------
		dict
			Mutated bounding box data.
		"""
		if not len(bounding['xMin']) == len(bounding['yMin']) == len(bounding['xMax']) == len(bounding['yMax']):
			raise AnnotationError()

		coordinates = []

		for k in range(len(bounding['xMin'])):
			coordinates.append(
				{
					'coordinates':
						{
							'x': Utils.calc_center(bounding['xMin'][k], bounding['xMax'][k]),
							'width': Utils.calc_dist(bounding['xMin'][k], bounding['xMax'][k]),
							'y': Utils.calc_center(bounding['yMin'][k], bounding['yMax'][k]),
							'height': Utils.calc_dist(bounding['yMin'][k], bounding['yMax'][k])
						},
					'label': bounding['label']
				})

		return coordinates


class TCHandler(object):
	def __init__(self):
		pass

	@staticmethod
	def attach_images(sf, image_dir):
		"""
		Joins an SFrame of images with SFrame of annotation data.

		Both SFrames share the column 'path', and are joined in a SQL-like manner by that column.

		Parameters
		----------
		sf: SFrame
			An SFrame representation of the data. Includes two columns: path, annotations.
		image_dir: str
			The folder path containing the images in JPEG format. Includes two columns: path, images.

		Returns
		-------
		SFrame
			An SFrame representation of the data with the images. Includes three columns: path, images, annotations.
		"""
		sf_images = tc.image_analysis.load_images(image_dir)
		return sf.join(sf_images, on='path', how='left')

	@staticmethod
	def write(sf, output_dir, filename='ig02'):
		"""
		Writes the SFrame to an output directory with filename.

		Parameters
		----------
		sf: SFrame
			An SFrame representation of the data. Includes three columns: path, images, annotations.
		output_dir: str
			The directory to write the SFrame to.
		filename: str, optional
			The name of the SFrame file.
		"""
		sf.save(os.path.join(output_dir, '{0}.sframe'.format(filename)))


def createSFrame(annotatations_dir=None, image_dir=None, output_dir=None):
	"""
	Parses through the annotations, refactors the data, and then creates the SFrame.

	The user can use this to parse through ImageNet format data and refactor them into a TuriCreate friendly form.
	If the user doesn't input all the relevant directory information, then this will prompt the user to enter it manually in runtime.

	Parameters
	----------
	annotatations_dir : str, optional
		The directory where the annotations are stored.
	image_dir : str, optional
		The directory where the images are stored.
	output_dir : str, optional
		The directory to write the SFrame files in.
	"""

	if not annotatations_dir or not image_dir or not output_dir:
		annotatations_dir = input("Annotations Directory: ").strip()
		image_dir = input("Images Directory: ").strip()
		output_dir = input("Output Directory: ").strip()

	handler = ImageHandler()

	finalData = {
		'path': [],
		'annotations': []
	}

	for obj_label in os.listdir(annotatations_dir):
		obj_path = os.path.join(annotatations_dir, obj_label)
		prefix = '({0}) {1}/{2} files:'
		if os.path.isdir(obj_path):
			num_files = len([x for x in os.listdir(obj_path) if x.endswith('.xml')])
			i = 0
			Utils.showProgress(0, num_files, prefix=prefix.format(obj_label, 0, num_files), length=50)

			for file in os.listdir(obj_path):
				if file.endswith('.xml'):
					xml_file = os.path.join(obj_path, file)
					xml_tree = et.parse(xml_file)
					data = handler.parse_xml(obj_label, xml_tree, image_dir)

					finalData['path'].append(data['path'])
					finalData['annotations'].append(data['annotations'])

					# Update the progress bar
					i += 1
					Utils.showProgress(i, num_files, prefix=prefix.format(obj_label, i, num_files), length=50)

	tcHandler = TCHandler()
	sf = tc.SFrame(finalData)
	sf = tcHandler.attach_images(sf, image_dir)
	tcHandler.write(sf, output_dir, 'original')


def explore(sframe_dir=None, draw_bounding_boxes=True, limit=10):
	"""
    Allows the user to explore by inspection the created SFrame.

    Parameters
    ----------
    sframe_dir : str, optional
        The path of the SFrame file.
    draw_bounding_boxes : bool, optional
        Whether or not to display the images with the bounding boxes drawn (good for manual inspection).
    limit : int, optional
        How many entries to display in the Turi Create Visualizer.
    """
	if not sframe_dir:
		sframe_dir = raw_input("Sframe Directory: ")
	sframe_dir = sframe_dir.strip()

	sf = tc.SFrame(sframe_dir)
	if draw_bounding_boxes:
		sf['image_with_ground_truth'] = tc.object_detector.util.draw_bounding_boxes(sf['image'], sf['annotations'])
		sf['image_with_ground_truth'].apply(lambda x: x)
	sf[:limit].explore()
