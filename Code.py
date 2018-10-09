from sklearn.cluster import MiniBatchKMeans
import numpy as np 
import cv2
import imutils

image=cv2.imread("/home/chetan/Documents/quantize/face.JPG")
(h,w)=image.shape[:2]
print(h,w)
image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
(h1,w1)=image.shape[:2]
image=image.reshape((h1*w1),3)
clt=MiniBatchKMeans(n_clusters=8)
labels=clt.fit_predict(image)
quant=clt.cluster_centers_.astype('uint8')[labels]
#print('l',labels)
#print('cluster',clt.cluster_centers_)
#print('q',quant)
#print(type(labels))
#print(type(clt.cluster_centers_))
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
z=imutils.resize(quant,width=500)
cv2.imshow("image", z)
cv2.imwrite('quant.jpg',z)
cv2.waitKey(0)
