"""Functions for reading youtube face data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageDraw
import numpy as np
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

rootDir = 'C:\\frame_images_DB\\frame_images_DB'
IMAGE_SIZE = 160

def resize_image(image):
	shape = image.size
	height = IMAGE_SIZE
	width = IMAGE_SIZE
	if shape[0] >= shape[1]:
		height = math.floor(shape[1] * IMAGE_SIZE / shape[0])
	else:
		width = math.floor(shape[0] * IMAGE_SIZE / shape[1])
	start_y = math.floor((IMAGE_SIZE - height) / 2)
	start_x = math.floor((IMAGE_SIZE - width) / 2)
	output = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3])
	output[start_y : start_y + height, start_x : start_x + width, :] = np.array(image.resize((width, height), Image.BILINEAR))
	return output

def make_box(x, y, size, image):
	shape = image.size
	height = IMAGE_SIZE
	width = IMAGE_SIZE
	ratio = 0
	if shape[0] >= shape[1]:
		height = math.floor(shape[1] * IMAGE_SIZE / shape[0])
		ratio = IMAGE_SIZE / shape[0]
	else:
		width = math.floor(shape[0] * IMAGE_SIZE / shape[1])
		ratio = IMAGE_SIZE / shape[1]
	start_y = math.floor((IMAGE_SIZE - height) / 2)
	start_x = math.floor((IMAGE_SIZE - width) / 2)
	return [(x - size/2) * ratio + start_x, (x + size/2) * ratio + start_x, (y - size/2) * ratio + start_y, (y + size/2) * ratio + start_y]

class DataSet(object):

	def __init__(self, filelist):
		fo = open(filelist, 'r')
		self._lines = fo.readlines()
		self._num_examples = len(self._lines)
		self._iter = 0

	def _load_image(self, index):
		parts = self._lines[index].split(',')
		image = Image.open(rootDir + '\\' + parts[0])
		resized_image = resize_image(image).reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3])
		box = np.array(make_box(float(parts[1]), float(parts[2]), float(parts[3]), image))
		image.close()
		return resized_image, box

	def next_batch(self, size):
		data = np.zeros([size, IMAGE_SIZE, IMAGE_SIZE, 3])
		truth = np.zeros([size, 4])
		for i in xrange(size):
			data[i, :, :, :], truth[i, :] = self._load_image(self._iter)
			self._iter = (self._iter + 13) % self._num_examples
		return data, truth

	@property
	def num_examples(self):
		return self._num_examples
