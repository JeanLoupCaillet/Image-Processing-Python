#! /usr/bin/env python

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from cv2 import calcHist 
from PIL import Image
from numpy.matlib import repmat as rep

Img=cv2.imread('Images/katrina-08-28-2005.jpg',0)
#Upper and Lower gray level
Data_up=[]
Data_lo=[]


# Upper Threshold
# Store the values of the future mask on a variable
def Up(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		Data_up.append(Img[x,y])

cv2.namedWindow('Define Upper Threshold', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Define Upper Threshold',Up)

while(1):
    cv2.imshow('Define Upper Threshold',Img)
    if cv2.waitKey(20) & 0xFF == 27: # Waiting for 'Echap' Key
        break

# Array or Scalar
X_up=min(Data_up)*1
Y_up=max(Data_up)*1

cv2.destroyAllWindows()

# Lower Threshold
# Store the values of the future mask on a variable
def Lower(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		Data_lo.append(Img[x,y])

cv2.namedWindow('Define Lower Threshold', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Define Lower Threshold',Lower)


while(1):
    cv2.imshow('Define Lower Threshold',Img)
    if cv2.waitKey(20) & 0xFF == 27: # Waiting for 'Echap' Key
        break

# Array or Scalar
X_lo=min(Data_lo)*1
Y_lo=max(Data_lo)*1



# Masks
Mask_up= cv2.inRange(Img, X_up, Y_up)
Mask_lo=cv2.inRange(Img, X_lo, Y_lo)


Nb_Pixel_Black_up=(np.sum(Mask_up)/255)
Nb_Pixel_Black_lo=(np.sum(Mask_lo)/255)

#plotting mask
cv2.destroyAllWindows()
cv2.imshow('Up',Mask_up)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Lo',Mask_lo)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Up.png',Mask_up)
cv2.imwrite('lo.png',Mask_lo)


# Results
print "Size of the picture: ", np.size(Img)
print 'Number of Black pixels with upper mask: ',Nb_Pixel_Black_up
print 'Number of Black pixels with lower mask: ',Nb_Pixel_Black_lo

##########################################################################################################
############################################ Other analysis ##############################################
##########################################################################################################


imgB = cv2.imread('Images/Av.png',0)
edgesB = cv2.Canny(imgB,100,200)
#Before
plt.figure(1)
plt.subplot(121),plt.imshow(imgB,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edgesB,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])



imgA=cv2.imread('Images/Ap.png',0)
edgesA = cv2.Canny(imgA,100,200)
#After
plt.figure(2)
plt.subplot(121),plt.imshow(imgA,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edgesA,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])



#Comparaison
plt.figure(3)
plt.subplot(121),plt.imshow(edgesB,cmap='gray')
plt.title('Edges Image Before Irma'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(edgesA,cmap='gray')
plt.title('Edges Image After Irma'),plt.xticks([]),plt.yticks([])




#Difference
Difference=cv2.subtract(edgesB,edgesA)
Result=not np.any(Difference) # if Difference is all zeros it will return False
if Result is True:
	print "The images are identical"
else:
	plt.figure(4)
	plt.imshow(Difference,cmap='gray')



#Histograms
plt.figure(5)
img = cv2.imread('Images/maria092017_Goes.jpg') # change image= change threshold

color = ('r','g','b')
for i,col in enumerate(color):
    histr = calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])


kat=cv2.imread('Images/kat290805.jpg')
plt.figure(6) 
plt.imshow(np.uint8(kat))

# Tresholding & red mask
lx=np.size(kat, axis=0)
ly=np.size(kat, axis=1)

# Katrina
lower_red=0
upper_red=150

# Maria
# lower_red=13
# upper_red=128


mask_red=(kat[:,:,0]>=lower_red) & (kat[:,:,0]<=upper_red) # Change kat into img if needed

mask_red=mask_red+0.

imgfr=np.zeros((lx, ly, 3))
imgfr[:,:,0]=np.uint8(np.multiply(np.double(kat[:,:,0]),np.double(mask_red)))
imgfr[:,:,1]=np.uint8(np.multiply(np.double(kat[:,:,1]),np.double(mask_red)))
imgfr[:,:,2]=np.uint8(np.multiply(np.double(kat[:,:,2]),np.double(mask_red)))


plt.figure(7)    
plt.imshow(imgfr)
plt.title('Mask applyed')

plt.figure(8)
#Maria
imgf=np.copy(kat)
for i in range(lx):
    for j in range(ly):
        if imgfr[i,j,0] == 0: 
               imgf[i,j,0]=imgfr[i,j,0]*0
               imgf[i,j,1]=imgfr[i,j,0]*0
               imgf[i,j,2]=imgfr[i,j,0]*0
plt.imshow(imgf)

#imgf=np.copy(kat)
#for i in range(lx):
#    for j in range(ly):
#        if imgfr[i,j,0] <> 0: 
#              imgf[i,j,0]=imgfr[i,j,0]
#              imgf[i,j,1]=imgfr[i,j,0]
#              imgf[i,j,2]=imgfr[i,j,0]
#
#plt.imshow(imgf)

somme_px_noir=np.sum(imgfr[:,135:968]==0)
print'Number of Black pixels : ',somme_px_noir
plt.show()
cv2.waitKey(0)

#plt.close('all')

################################################################################################
############################################ Knn ###############################################
################################################################################################



#Blank=[]
#kat=cv2.imread('Images/kat290805.jpg')
#im=Image.open('Images/kat290805.jpg')
#[L,C]=im.size
#
#
#d=3 # 3 layers
#def White(event,x,y,flags,param):
#	if event == cv2.EVENT_LBUTTONDBLCLK:
#		Blank.append(kat[x,y])
#
#cv2.namedWindow('Define Data Blank', cv2.WINDOW_AUTOSIZE)
#cv2.setMouseCallback('Define Data Blank',White)
#
#while(1):
#    cv2.imshow('Define Data Blank',kat)
#    if cv2.waitKey(20) & 0xFF == 27: # Waiting for 'Echap' Key
#        break
#    
#    
#cv2.destroyAllWindows()
## Check
## print'Blank values= ',Blank 
#
#Mask=np.zeros((L,C))
#NbData=np.size(Blank)/d
## Algorithm
#
#for c in range (C):
#    for l in range (L):
#        R[:,:,0]=kat[l,c,0]
#        G[:,:,1]=np.uint8(kat[l,c,1])
#        B[:,:,2]=np.uint8(kat[l,c,2])
#        RGB=rep([R, G, B],NbData,1)
#        Dist=np.power((RGB-Blank),2)
#        Mask[l,c]=1; # Put Black
#
#cv2.imshow('Mask by Knn',Mask)
