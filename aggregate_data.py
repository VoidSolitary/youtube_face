import os
import random

rootDir = 'D:\\Downloads\\frame_images_DB\\frame_images_DB'
trainFile = open('train.txt', 'w')
testFile = open('test.txt', 'w')
trainCount = 0
testCount = 0
fileCount = 0
filelist = os.listdir(rootDir)
for filename in filelist:
	if not filename.endswith('.txt'):
		fileCount = fileCount + 1
		if fileCount % 100 == 0:
			print('Processing %d' % fileCount)
		fo = open(rootDir + '\\' + filename + '.labeled_faces.txt', 'r')
		isTest = (random.randint(0, 10) == 0)
		lineCount = 0
		for line in fo.readlines():
			if lineCount % 10 == 0:
				parts = line.split(',')
				if isTest:
					testCount = testCount + 1
					testFile.write('%s,%s,%s,%s\n' % (parts[0], parts[2], parts[3], parts[4]))
				else:
					trainCount = trainCount + 1
					trainFile.write('%s,%s,%s,%s\n' % (parts[0], parts[2], parts[3], parts[4]))
			lineCount = lineCount + 1
		fo.close()
print('Number of subdirs: %d' % fileCount)
print('Number of training images: %d' % trainCount)
print('Number of test images: %d' % testCount)

trainFile.close()
testFile.close()