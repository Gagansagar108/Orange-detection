import cv2
import numpy as np

threat_image=cv2.imread(r'E:\New folder\BaggageAI_CV_Hiring_Assignment\threat_images\t3.jpg')
bg_image=cv2.imread(r'E:\New folder\BaggageAI_CV_Hiring_Assignment\background_images\b5.jpg')
def get_mask(img):      #mask to seperate the object and background from image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(img)  #splitting the channels
    white=np.zeros(img.shape).astype('uint8')
    dark_image=np.zeros(img.shape[:2]).astype('uint8')  #a blank image
    for x in range(blue.shape[0]):  #finding out the darker pixel so that thesholding can be more effective and accurate
        for y in range(blue.shape[1]):
            dark_image[x][y]=min(blue[x][y],green[x][y],red[x][y])
    g_blurred = cv2.GaussianBlur(dark_image, (5, 5), 0) # blurring to remove noises and make smooth
    r,binary=cv2.threshold(g_blurred,225,255,cv2.THRESH_BINARY_INV)
    mask=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,(3,3),iterations=3)  #some morphological operations to remove error

    return mask

def get_corner_points(mask):    # the extreme points of the object
    cnts,h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finding contours
    c = max(cnts, key=cv2.contourArea)  # contour with maxrea
    left = tuple(c[c[:, :, 0].argmin()][0])    #(x,y) points of lestmost point on object
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    return left,top,right,bottom

def get_centroids(mask):  #returns the centroids of the object
    cnts,h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # compute the center of the contour by
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return  cX,cY


def get_radius_of_max_circle_inscribed(mask,cx,cy):  #it returns the max circles that can be incribed by the background object

    cnts, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    distance = []
    for x in c:   #calculating distance of the centroids with each contour on edge
            dist = int(np.sqrt(np.square(x[0][0] - cx) + np.square(x[0][1] - cy)))
            distance.append(dist)
    return min(distance)   #return the radius of the desired circle

def rotate_image(image, angle):  #ratate the image anti clockwise wrt centroid

  center = (np.array(image.shape[1::-1])/2)
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat,image.shape[1::-1],borderValue=(255,255,255), flags=cv2.INTER_LINEAR)
  return result

#resizing the image such that we dont loose the object data after rotating it
rows=max(threat_image.shape[:])
white=(np.ones((rows,rows,3))*255).astype('uint8')
print(threat_image.shape,white.shape)
white[:,200:threat_image.shape[1]+200,]=threat_image[:,:,:]   #adding some whitespace along the coloumns of threat_image ,if the object is rotated wwe dont lose the area of object
threat_image=white



mask=get_mask(threat_image)

left,top,right,bottom= get_corner_points(mask)   #extreme points of the mask

###just to illustrate drawing corner points on the object
""" 
cv2.circle(threat_image,left,3,(255,0,0),2)
cv2.circle(threat_image,top,3,(0,0,255),2)
cv2.circle(threat_image,right,3,(0,255,0),2)
cv2.circle(threat_image,bottom,3,(0,0,0),2)
p1=(left[0],top[1])     #pi and p2 are the points of rectangle which entirely encloses the object
p2=(right[0],bottom[1])
#rect=cv2.rectangle(threat_image,p1,p2,255,2)  """   #draws the rectangle over the object
#
cropped_threat=threat_image[top[1]-80:bottom[1]+80,left[0]-80:right[0]+80]     #crops the object , which left behind the unneccesory background from object
rotated=rotate_image(cropped_threat,45)

def overlay(bg_img,threat):  #paste the object over background
        cx, cy = get_centroids(get_mask(bg_img))  #gets the centroids of the background image
        bg_mask=get_mask(bg_img)    #creates the mask of the background image
        r=get_radius_of_max_circle_inscribed(bg_mask,cx,cy)   #gets the radius of the max circle which is not exceeding the backgound
        side = int(np.sqrt(2)*r*0.8)    #max requied side of rectangle ,for safety taking 80% of the max side allowed
        ratio = side/max(threat.shape[0],threat.shape[1])
        y = int(threat.shape[0] * ratio)
        x = int(threat.shape[1] * ratio)
        gray=cv2.cvtColor(threat,cv2.COLOR_BGR2GRAY)
        ones = np.ones((bg_img.shape[0], bg_img.shape[1])).astype('uint8')*255
        image_1 = np.dstack([bg_img, ones])
        l=cx-int(x/2)
        r=cx+int(x/2)
        t=cy-int(y/2)
        b=cy+int(y/2)

        threat=cv2.resize(threat,((r-l),(b-t)))    #resizing the object to fit inside the circle
        img_a = cv2.cvtColor(threat, cv2.COLOR_BGR2BGRA)     #calculating alpha
        mask = get_mask(threat)
        _, alpha = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
        alpha = alpha / 255.0
        alpha2 = 1 - alpha
        for c in range(0, 3):    #overlaying object on backgound
            image_1[t:b,l:r,c]=(alpha2*image_1[t:b,l:r,c]) + (alpha*img_a[:,:,c])
        return  image_1

final=overlay(bg_image,rotated)
cv2.imshow('final',final)

final_bgr=cv2.cvtColor(final,cv2.COLOR_BGRA2BGR)

added_weight=cv2.addWeighted(bg_image,0.4,final_bgr,0.6,0)   #making it translucent
cv2.imshow('finakkkl',added_weight)
cv2.imwrite(r'E:\New folder\result_5_4.jpg',added_weight)
cv2.waitKey(0)

