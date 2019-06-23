import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import cv2
import os
from xmlToSFrame import ImageHandler, TCHandler
import turicreate as tc
from utils import Utils


class DataAugHandler(object):
	# light augmentation sequence
	ia.seed(1)

	seq = iaa.Sequential([
		iaa.Fliplr(0.5),  # horizontal flips
		iaa.Crop(percent=(0, 0.1)),  # random crops
		# Small gaussian blur with random sigma between 0 and 0.5.
		# But we only blur about 50% of all images.
		iaa.Sometimes(0.5,
					  iaa.GaussianBlur(sigma=(0, 0.5))
					  ),
		# Strengthen or weaken the contrast in each image.
		iaa.ContrastNormalization((0.75, 1.5)),
		# Add gaussian noise.
		# For 50% of all images, we sample the noise once per pixel.
		# For the other 50% of all images, we sample the noise per pixel AND
		# channel. This can change the color (not only brightness) of the
		# pixels.
		iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
		# Make some images brighter and some darker.
		# In 20% of all cases, we sample the multiplier once per channel,
		# which can end up changing the color of the images.
		iaa.Multiply((0.8, 1.2), per_channel=0.2),
		# Apply affine transformations to each image.
		# Scale/zoom them, translate/move them, rotate them and shear them.
		iaa.Affine(
			scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
			translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
			rotate=(-25, 25),
			shear=(-8, 8)
		)
	], random_order=True)  # apply augmenters in random order

	def __init__(self, sframe_dir):
		self.sframe_dir = sframe_dir
		self.dataSframe = self.getSFrame()

	def getSFrame(self):
		for file in os.listdir(self.sframe_dir):
			if file.endswith('.sframe'):
				data = tc.SFrame(os.path.join(self.sframe_dir, file))
				self.originalSF = data
		return data

	def createAugmentedandFinalSFrameFile(self, augImages_dir):
		augmented_sf = self.getAugmentationsSFrame(augImages_dir)
		# print(sf)
		# TCHandler.write(sf, self.sframe_dir, 'igAug')
		self.originalSF.append(augmented_sf)
		TCHandler.write(augmented_sf, self.sframe_dir, 'augmented')
		TCHandler.write(self.originalSF, self.sframe_dir, 'final')

	def getAugmentationsSFrame(self, augImages_dir):
		data = {
			'annotations': [],
			'path': []
		}

		prefix = '{0}/{1} images augmented:'
		Utils.showProgress(0, len(self.dataSframe), prefix=prefix.format(0, len(self.dataSframe)), length=50)
		count = 0

		for row in self.dataSframe:
			img = cv2.imread(row["path"])
			bbs = self.getBoundingBoxesOnImage(row["annotations"], img)
			image_points_aug, bbs_aug = self.seq(image=img, bounding_boxes=bbs)
			cv2.imwrite(os.path.join(augImages_dir, '{0}.JPEG'.format(count)), image_points_aug)

			data['path'].append(os.path.join(augImages_dir, '{0}.JPEG'.format(count)))
			count += 1
			Utils.showProgress(count, len(self.dataSframe), prefix=prefix.format(count, len(self.dataSframe)),
							   length=50)

			if len(bbs_aug.remove_out_of_image().clip_out_of_image().to_xyxy_array()) != 0:
				bounding = {
					'label': row["annotations"][0]['label'],
					'xMin': [],
					'xMax': [],
					'yMin': [],
					'yMax': [],
				}

				for bb in bbs_aug.remove_out_of_image().clip_out_of_image().to_xyxy_array():
					# bb is (x1, y1, x2, y2)
					# bounding['label'] = bb.label
					bounding['xMin'].append(bb[0])
					bounding['xMax'].append(bb[2])
					bounding['yMin'].append(bb[1])
					bounding['yMax'].append(bb[3])

				data['annotations'].append(ImageHandler.transform_bounding(bounding))
			else:
				data['annotations'].append([])

		augSFrame = tc.SFrame(data)
		sf_images = tc.image_analysis.load_images(augImages_dir, with_path=True)
		return augSFrame.join(sf_images, on='path', how='left')

	def visualizeOriginal(self, aug_dir):

		count = 0
		for row in self.dataSframe:
			img = cv2.imread(row["path"])
			bbs = self.getBoundingBoxesOnImage(row["annotations"], img)
			image_after = self.draw_bbs(img, bbs, 100)
			cv2.imwrite(os.path.join(aug_dir, '{0}.JPEG'.format(count)), image_after)
			count += 1
			print(count)

	def getBoundingBoxesOnImage(self, annotations, img):
		bbsList = []
		for annotation in annotations:
			bbsList.append(self.getBoundingBoxFromAnnotation(annotation))
		return BoundingBoxesOnImage(bbsList, img.shape)

	def getBoundingBoxFromAnnotation(self, annotation):

		centerX = annotation['coordinates']['x']
		centerY = annotation['coordinates']['y']
		width = annotation['coordinates']['width']
		height = annotation['coordinates']['height']
		x1 = centerX - width / 2
		x2 = centerX + width / 2
		y1 = centerY - height / 2
		y2 = centerY + height / 2
		label = annotation['label']
		return BoundingBox(x1, y1, x2, y2, label)

	GREEN = [0, 255, 0]
	ORANGE = [255, 140, 0]
	RED = [255, 0, 0]

	# Pad image with a 1px white and (BY-1)px black border
	def pad(self, image, by):
		if by <= 0:
			return image
		image_border1 = np.pad(
			image, ((1, 1), (1, 1), (0, 0)),
			mode="constant", constant_values=255
		)
		image_border2 = np.pad(
			image_border1, ((by - 1, by - 1), (by - 1, by - 1), (0, 0)),
			mode="constant", constant_values=0
		)

		return image_border2

	def draw_bbs(self, image, bbs, border):
		image_border = self.pad(image, border)
		for bb in bbs.bounding_boxes:
			if bb.is_fully_within_image(image.shape):
				color = self.GREEN
			elif bb.is_partly_within_image(image.shape):
				color = self.ORANGE
			else:
				color = self.RED
			image_border = bb.shift(left=border, top=border) \
				.draw_on_image(image_border, size=2, color=color)

		return image_border


def augmentData(sframe_dir, augImages_dir):
	handler = DataAugHandler(sframe_dir)
	handler.createAugmentedandFinalSFrameFile(augImages_dir)
