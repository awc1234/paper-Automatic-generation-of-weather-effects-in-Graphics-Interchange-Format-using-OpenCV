# -*- coding: utf-8 -*-
"""
@author: GUser
"""

import os,sys
import cv2
from matplotlib import pyplot as plt
import numpy as np

import datetime
import imageio
from pprint import pprint
import time
e=sys.exit

num_path = 20#30
kkk = 0
THRESHOLD_VALUE = 10000
 
#####rain effect

'''
    generate_random_lines
    : select rain drop pixel using random values which satisfies selected_pixel.

'''
def generate_random_lines(imshape,slant,drop_length):

    rows, cols = imshape[:2]
    drops=[]
    for i in range(1500): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,rows)
        else:
            x= np.random.randint(0,rows-slant)

        y= np.random.randint(0,cols-drop_length)

        if x < rows and y < cols:
            if selected_pixel[x][y] == 1 or selected_pixel[x][y] == 2:
                 drops.append((y,x))
    return drops
        
'''
    add_rain
    : add rain effect to generated rain drop pixel by generate_random_lines.

'''
def add_rain(image):
       
    imshape = image.shape
    #slant is used for rain slope.
    slant_extreme=1
    slant= np.random.randint(-slant_extreme,slant_extreme) 
    drop_length=20
    drop_width=1
    #you can use it for rain effect.
    drop_bright=50
    #drop_color=(200,200,200) ## a shade of gray
    rain_drops= generate_random_lines(imshape,slant,drop_length)

    for rain_drop in rain_drops:
        rd = ((int(image[rain_drop[1]][rain_drop[0]][0])+drop_bright ,int( image[rain_drop[1]][rain_drop[0]][1])+drop_bright, int(image[rain_drop[1]][rain_drop[0]][2])+drop_bright ))
        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),rd,drop_width)
 
    #if you want blur, use this line.
    #image= cv2.blur(image,(7,7)) ## rainy view are blurry
    
    brightness_coefficient = 0.7 ## rainy days are usually shady 
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    
    return image

def show_selected_pixel(img):
    img_s = np.copy(img)
    rows, cols = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            if selected_pixel[i][j]==1:#line
                img_s[i,j] = (255,0,0)
            if selected_pixel[i][j]==2:#dnc
                img_s[i,j] = (0,255,0)
    return img_s


def show_path(img, selected_p):
    img_show_path = np.copy(img)
    x_coords, y_coords = np.transpose([(i, int(j)) for i,j in enumerate(selected_p)])
    img_show_path[x_coords, y_coords] = (255,0,0)
    return img_show_path



def evaluete_pvalue(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    kernel = np.ones((3,3), np.float32)
    kernel[0,0] = kernel[0,1] = kernel[0,2] = 10/60
    kernel[1,1] = kernel[2,0] = kernel[2,1] = kernel[2,2] = 3/60
    kernel[1,0] = kernel[1,2] = 9/60
    
    rain= cv2.convertScaleAbs(cv2.filter2D(gray,-1,kernel))
    X_sobel = cv2.convertScaleAbs(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3))
    Y_sobel = cv2.convertScaleAbs(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3))
    Preprocessing = (cv2.addWeighted(X_sobel, 0.5, Y_sobel,0.5,0))
    return cv2.addWeighted(Preprocessing, 0.99, rain,0.01,0) 

def find_path(img, V_pixel):
    
    rows, cols = img.shape[:2]
    selected_p = np.zeros(img.shape[0])
    D = np.zeros(img.shape[:2]) + sys.maxsize #[rows x cols]
    D[0,:] = np.zeros(img.shape[1])
    path = np.zeros(img.shape[:2]) # rows x cols 2-dim array
    
    for row in range(rows-1):
       for col in range(cols):
           for t in range(-2,2):
               ncol=col+t
               if 0<= ncol and ncol<cols:#using -1 only instead of -1 ~ -3
                   if D[row+1, ncol] > D[row,col]+V_pixel[row+1, ncol]:
                       D[row+1, ncol] = D[row,col]+V_pixel[row+1, ncol]
                       path[row+1, ncol] = col
                   
    selected_p[rows-1] = np.argmin(D[rows-1, :])
    
    for i in (x for x in reversed(range(rows)) if x > 0):    
        selected_pixel[original_xy[i][int(selected_p[i])][0]][original_xy[i][int(selected_p[i])][1]] = 1
        selected_p[i-1] = path[i, int(selected_p[i])]

    return selected_p


def erase_path(image, selected_p):
    rows, cols = image.shape[:2]
  
    for i in range(rows):
        for j in range(int(selected_p[i]), cols-1):
            image[i,j] = image[i, j+1]
            original_xy[i][j] = (i, j+1)
    
    image = image[:, 0:cols-1]
    return image


def find_selected_pixel(sx, sy, ex, ey):
    cnt = 0
    for i in range(sy, ey):
        for j in range(sx, ex):#fixed 5/30 for j in range(ey.ex)
            if selected_pixel[i][j]==1:
                cnt = cnt + 1
    return cnt

def check_quadrant_in_object(sx, sy, ex, ey):#fixed 5/30 it was empty 

    for i in range(sy, ey):
        for j in range(sx, ex):
            if ws_img[i][j] != 255:
                return False
    return True

def key_pixel(t):
    return t[0]

def add_selected_pixel(sx, sy, ex, ey, num_should_add):
    if sx >= ex or sy >= ey: return
    if int(num_should_add) <=0: return
    
    threshold = THRESHOLD_VALUE
    
    list_of_pixel=[]
    if (ex-sx+1)*(ey-sy+1) <= 9 or num_should_add<=9:
        for i in range(sy, ey):
            for j in range(sx, ex):
                if selected_pixel[i][j]==0: 
                    list_of_pixel.append((original_pvalue[i,j],i,j))
        t=0
        list_of_pixel.sort(key=key_pixel)
        
        for pixel in list_of_pixel:
            t = t + 1
            selected_pixel[pixel[1]][pixel[2]] = 2
            if t<= num_should_add: break
        return
    
    mid_x = int((sx+ex)/2)
    mid_y = int((sy+ey)/2)

   
    qu = [(sx, sy, mid_x, mid_y), 
	       (mid_x+1, sy, ex, mid_y),
	       (sx, mid_y+1, mid_x, ey),
	       (mid_x+1, mid_y+1, ex, ey)]
 
    cnt_qu = 0
    for q in qu:
        if find_selected_pixel(q[0],q[1],q[2],q[3]) < threshold:
            cnt_qu = cnt_qu + 1
    for q in qu:
        num_selected_pixel = find_selected_pixel(q[0],q[1],q[2],q[3])
        if num_selected_pixel < threshold and check_quadrant_in_object(q[0],q[1],q[2],q[3])==False:
            add_selected_pixel(q[0],q[1],q[2],q[3], num_should_add/cnt_qu)
    return


'''
    create gif for jpgs.
'''

def create_gif(filenames, duration):
	images = []
	for filename in filenames:
		images.append(imageio.imread(filename))
	output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
	imageio.mimsave(output_file, images, duration=duration)


if __name__=='__main__':
         
    img_input = cv2.imread('test1.png')
    ##111 img_input

    img = np.copy(img_input)

    #this lines for generate object detection.
    for_cvt = np.copy(img_input)

    gray = cv2.cvtColor(for_cvt,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    orginal_img = np.copy(img_input)
    rows, cols = img.shape[:2]
    
    global selected_pixel
    global original_xy
    global ws_img# added 5/30
     
    selected_pixel = [[0]*cols for i in range(rows)]
    original_xy = [[(0,0)]*cols for i in range(rows)]
   
    ws_img = sure_bg

    for i in range(rows):
        for j in range(cols):
            original_xy[i][j] = (i,j)
    
    img_show_path = np.copy(img_input)

    pvalue = evaluete_pvalue(img)
    original_pvalue = evaluete_pvalue(orginal_img)
    

    cv2.imshow('1',orginal_img);
    
   # img = sure_bg
    for i in range(num_path):
        a_path = find_path(img, pvalue)
        img_show_path = show_path(img_show_path, a_path)
        img = erase_path(img, a_path)
        pvalue = evaluete_pvalue(img)
        print('Stage %d' %(i+1))
    
    
    cv2.imshow('2',orginal_img);
    
    add_selected_pixel(0, 0, int(cols-1), int(rows-1), 10000)
    
    # cv2.imshow('selected pixel',show_selected_pixel(orginal_img));
   
       
    
    for i in range(20):
        img = np.copy(img_input)
        add_rain(img)
        f_name = 'rain%d.jpg'%i
        cv2.imwrite(f_name,img)
    
    # cv2.imshow('1',orginal_img)
   
    duration = 0.2
    filenames = sorted(filter(os.path.isfile, [x for x in os.listdir() if x.endswith(".jpg")]), 
                key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(datetime.now().timetuple()))
 
    create_gif(filenames, duration)

   
    print("done!\n")
    #cv2.waitKey()
    #cv2.destroyAllWindows() 
    
    
