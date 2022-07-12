#!/usr/bin/env python
# coding: utf-8

# In[96]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[97]:


image = cv2.imread('ideal1.png')


# In[98]:


plt.imshow(image)


# In[99]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap="gray")


# In[100]:


blur = cv2.medianBlur(gray, 7)
plt.imshow(blur, cmap="gray")


# In[101]:


th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
plt.imshow(th, cmap="gray")


# In[102]:


blur2 = cv2.medianBlur(th, 7)
plt.imshow(blur2, cmap="gray")


# In[103]:


blur3 = cv2.medianBlur(blur2, 7)
plt.imshow(blur3, cmap="gray")


# In[104]:


kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(blur3, kernel, iterations=1)
dilation = cv2.dilate(blur3, kernel, iterations=1)
plt.imshow(erosion, cmap="gray")


# In[110]:


blur4 = cv2.medianBlur(erosion, 9)
plt.imshow(blur4, cmap="gray")


# In[106]:


kernel = np.ones((5,5), np.uint8)
erosion2 = cv2.erode(blur4, kernel, iterations=1)
dilation2 = cv2.dilate(blur4, kernel, iterations=1)
plt.imshow(erosion2, cmap="gray")


# In[111]:


output = image.copy()
circles = cv2.HoughCircles(blur4, cv2.HOUGH_GRADIENT, 1, 150, param1=50, param2=30, minRadius=30, maxRadius=200)
detected_circles = np.uint16(np.around(circles))
j=0
if circles is not None:
    for (x,y,r) in circles[0, :]:
        print(x , y, r)
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
        j+=1
plt.imshow(output, cmap="gray")
print(j)


# In[ ]:




