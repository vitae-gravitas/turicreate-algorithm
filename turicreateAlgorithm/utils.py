import sys
import os
import pickle

__author__ = "Matt Zhou"
__copyright__ = "Copyright 2019"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Matt Zhou"
__email__ = "mattzh1314@berkeley.edu"
__status__ = "Production"

class Utils(object):
	def __init__(self):
		pass

	@staticmethod
	def showProgress(iteration, total, prefix='Progress:', suffix='', decimals=1, length=100, fill='*'):
		"""
		Prints a progress bar.

		Parameters
		----------
		iteration : int
			Current interation.
		total : int
			Total iterations until completion.
		prefix : str, optional
			Prefix string.
		suffix : str, optional
			Suffix string.   
		decimals : int, optional
			Positive number of decimals in percent complete.
		length : int, optional
			Character length of bar.
		fill : str, optional
			Bar fill character.
		"""
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		sys.stdout.write('%s |%s| %s%% %s\r' % (prefix, bar, percent, suffix))
		sys.stdout.flush()
		# Print New Line on Complete
		if iteration == total: 
			print()

	@staticmethod	
	def calc_center(c_min, c_max):
		"""
		Calculates the center point between two coordinate points.

		Parameters
		----------
		c_min : int
			The minimum coordinate.

		c_max : int
			The maximum coordinate.

		Returns
		-------
		int
			Center point between two coordinates.
		"""
		return c_min + (c_max - c_min) / 2

	@staticmethod
	def calc_dist(c_min, c_max):
		"""
		Calculates the distance between two coordinate points.

		Parameters
		----------
		c_min : int
			The minimum coordinate.

		c_max : int
			The maximum coordinate.

		Returns
		-------
		int
			Distance between two coordinates.
		"""

		return c_max - c_min

	@staticmethod
	def write_as_pckl(data, name, output_dir):
		"""
		Writes data to a directory as a pickle file.

		Parameters
		----------
		data : dict
			Data points of interest as key value pairs.
		name : str
			The output file name.
		output_dir : str
			The directory to write the file in.
		"""
		with open(os.path.join(output_dir, '{0}.pickle'.format(name)), 'wb') as f:
			pickle.dump(data, f)

	@staticmethod
	def make_dir(dir_path):
		"""
		Creates a directory if it doesn't already exist.

		Parameters
		----------
		dir_path : str
			The directory path to create.

		Returns
		-------
		bool
			True if successfully created a directory, false otherwise.
		"""
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)
			return True
		return False




