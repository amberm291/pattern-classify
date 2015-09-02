import cv2
import numpy as np
import glob
import re
#from featureextraction import featex
from sklearn import svm, preprocessing
import cPickle as pickle
from urllib2 import urlopen
import sys

def sift(img):
	grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	grayImg = cv2.resize(img,None,fx=0.25,fy=0.25)
	grayImg = grayImg.astype(np.uint8)  
	dense = cv2.FeatureDetector_create("Dense")
	sift = cv2.SIFT()
	kp = dense.detect(grayImg)
	kp,des = sift.compute(grayImg,kp)
	size = des.shape
	imgFeature = np.reshape(des,(1,size[0]*size[1]))
	return imgFeature

def hog(img):
	grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	hogDescrip = cv2.HOGDescriptor()
	grayImg = cv2.resize(grayImg,None,fx=0.25,fy=0.25)
	hogFeature = hogDescrip.compute(grayImg)
	imgFeature = hogFeature.T
	return imgFeature

def gabor(img):
	kern = cv2.getGaborKernel((31,31),3.85,np.pi/4,8.0, 1.0, 0, ktype=cv2.CV_32F)
	img = cv2.filter2D(img, cv2.CV_8UC3, kern)
	return img

def gauss(img):
	img = cv2.GaussianBlur(img,(5,5),0)
	return img

testPath = sys.argv[1]
strLabel = sys.argv[2]
params = open('config').read()
imgFilter, featDescrip = params.split("\t")

'''function to calssify an image'''
def classify(img):
	try:
		'''calculating number of lines present in the image'''
		#blur = cv2.GaussianBlur(img,(5,5),0)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray,50,150)
		lines = cv2.HoughLines(edges,1,np.pi/180,100)
		'''if no of lines less than 100, classify as solid'''
		if lines.shape[1] < 100:
			pattern = 3
			return pattern
			'''if no of lines greater than 100, we use the classifier'''
		else:
			'''start code block from here'''
			if imgFilter == 'None':
				imgFeature = eval(featDescrip+'(img)')
			else:
				filteredImg = eval(imgFilter+'(img)')
				imgFeature = eval(featDescrip+'(filteredImg)')

			'''till here'''
			scaler = pickle.load(open("scaler.p","rb"))
			fVec = scaler.transform(imgFeature)
			clf = pickle.load(open("ShirtsCheckVsStripeVsSolid.p","rb"))
			label = clf.predict(imgFeature)
			return label
	except Exception, e:
		print e
		return 0		

'''wrapper function'''
def pattern(**kwargs):
	
	url = kwargs.get('url','')
	path = kwargs.get('path','')
	flag = False
	if url:
		try:
			req = urlopen(url).read()
			arr = np.asarray(bytearray(req), dtype=np.uint8)
			img = cv2.imdecode(arr,-1)
			flag = True
		except Exception, e:
			print e
			pass

	if path:
		try:
			img = cv2.imread(path)
			flag = True
		except Exception, e:
			print e
			pass

	if flag == True:
		pattern = str(classify(img))
		patMap = {"[1]":"Checkered","[2]":"Striped","3":"Solid"}
		return patMap[pattern]


def main():
	files = glob.glob(testPath+'/*')
	label = np.zeros((2,len(files)))
	label[1,:] = int(strLabel)*np.ones(len(files))
	c = 0
	for i in xrange(len(files)):
		print "processing",files[i]
		img = cv2.imread(files[i])
		label[0,i] = classify(img)
		if label[0,i] == label[1,i] or label[0,i] == 0:
			c += 1
		print label[0,i]
	print float(c*100)/len(files)

if __name__ == "__main__":
	main()
