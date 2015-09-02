import numpy as np
import math
from sklearn import svm, preprocessing
import cPickle as pickle
import re
import glob
import cv2

digits = re.compile(r'(\d+)')
def tokenize(filename):
    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))

flag = True
hog = cv2.HOGDescriptor()
files = glob.glob("/Users/11162/GHC/ChecksVsStripesVsSolids/*")       #path to training images folder
files.sort(key = tokenize)	#sorting training files
files = files[0:2*len(files)/3]
'''extracting the training feature matrix'''
for i in range(0,len(files)):
	img = cv2.imread(files[i])
	''' Put code snippets from here'''
	
	
	grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	grayImg = cv2.resize(img,None,fx=0.25,fy=0.25)
	grayImg = grayImg.astype(np.uint8)  
	dense = cv2.FeatureDetector_create("Dense")
	sift = cv2.SIFT()
	kp = dense.detect(grayImg)
	kp,des = sift.compute(grayImg,kp)
	size = des.shape
	imgFeature = np.reshape(des,(1,size[0]*size[1]))
	img = cv2.GaussianBlur(img,(5,5),0)

	
	'''till here'''
	if flag:
		feature = imgFeature
		flag = False
	else:
		feature = np.vstack((feature, imgFeature))
	print "Processing",files[i]

'''scaling and normalizing features'''
scaler = preprocessing.MinMaxScaler()
scaler.fit(feature)
feature = scaler.transform(feature)
pickle.dump(scaler,open('scaler.p','wb'))
print feature.shape

g = list()

for i in range(0,len(files)/2):
	g.append(1)
for i in range(len(files)/2,len(files)):
	g.append(2)
'''training SVM'''
lin_clf = svm.LinearSVC(C=0.4)
lin_clf.fit(feature,g) 
pickle.dump( lin_clf, open("ShirtsCheckVsStripeVsSolid.p", "wb" ) )
#print lin_clf.support_vector_.shape

