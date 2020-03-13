#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import math
import numpy as np
import xml.etree.ElementTree as ET
import os
import argparse
import random
from PIL import Image, ImageEnhance, ImageChops


# In[ ]:


def rotate_image(src, angle, scale=1):
    w = src.shape[1]
    h = src.shape[0]
#     print(w, h)
    # 角度變弧度
    rangle = np.deg2rad(angle)  # angle in radians
#     print(rangle)
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
#     print(nw, nh)
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    dst = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
#     print(dst)
    # 仿射變換
    return dst, nw, nh


# In[ ]:


def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # 獲取旋轉後圖像的長和寬
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]                                   # rot_mat是最終的旋轉矩陣
    # point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))          #這種新畫出的框大一圈
    # point2 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
    # point3 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
    # point4 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
    point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))   # 獲取原始矩形的四個中點，然後將這四個點轉換到旋轉後的座標系下
    point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
    point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
    concat = np.vstack((point1, point2, point3, point4))            # 合併np.array
    # 改變array型別
    concat = concat.astype(np.int32)
    rx, ry, rw, rh = cv2.boundingRect(concat)                        #rx,ry,為新的外接框左上角座標，rw為框寬度，rh為高度，新的xmax=rx+rw,新的ymax=ry+rh
    return rx, ry, rw, rh


# In[ ]:


def angel(imgpath, xmlpath, amax=24, amin=1):
    xml_path = xmlpath         #源影象路徑
    img_path = imgpath         #源影象所對應的xml檔案路徑
#     rotated_imgpath = './datasets/DA_test/'
#     rotated_xmlpath = './datasets/DA_test/'
    angel_random = random.randint(amin, amax)
    for angel in [angel_random*15]:
#     for angle in (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345):
#         for i in os.listdir(img_path):
#             a, b = os.path.splitext(i)                            #分離出檔名a
#         img = cv2.imread(img_path + a + '.jpg')
        img = cv2.imread(img_path)
        rotated_img = rotate_image(img,angel)
#             cv2.imwrite(rotated_imgpath + a + '_'+ str(angle) +'d.jpg',rotated_img[0])
#             print (str(i) + ' has been rotated for '+ str(angle)+'°')
#         tree = ET.parse(xml_path + a + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        root.iter('filename')
        
        xminl = []
        yminl = []
        xmaxl = []
        ymaxl = []
        
        for box in root.iter('bndbox'):
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            x, y, w, h = rotate_xml(img, xmin, ymin, xmax, ymax, angel)

            xminl.append(x)
            yminl.append(y)
            xmaxl.append(x+w)
            ymaxl.append(y+h)
            # cv2.rectangle(rotated_img, (x, y), (x+w, y+h), [0, 0, 255], 2)   #可在該步驟測試新畫的框位置是否正確
            # cv2.imshow('xmlbnd',rotated_img)
            # cv2.waitKey(200)
#             box.find('xmin').text = str(x)
#             box.find('ymin').text = str(y)
#             box.find('xmax').text = str(x+w)
#             box.find('ymax').text = str(y+h)
        for img_size in root.iter('size'):
#                 print(img_size.find('width').text)
            img_size.find('width').text = str(int(rotated_img[1]+1))
#                 print(img_size.find('height').text)
            img_size.find('height').text = str(int(rotated_img[2]+1))
#             tree.write(rotated_xmlpath + a + '_'+ str(angle) +'d.xml')
#             print (str(a) + '.xml has been rotated for '+ str(angle)+'°')
    return rotated_img, xminl, yminl, xmaxl, ymaxl, str(int(rotated_img[1]+1)), str(int(rotated_img[2]+1)), 'a'+str(angel_random*15)


# In[ ]:


def hue(img_raw, amax=5, amin=0.1):
#     img_raw = Image.open(img_path)
    cn = random.randint(amin*10, amax*10)
    imgcn = ImageEnhance.Color(img_raw).enhance(cn/10)
    return imgcn, 'h'+str(cn)


# In[ ]:


def contrast(img_raw, amax=2, amin=0.8):
#     img_raw = Image.open(img_path)
    cn = random.randint(amin*10, amax*10)
    imgcn = ImageEnhance.Contrast(img_raw).enhance(cn/10)
    return imgcn, 'c'+str(cn)


# In[ ]:


def brightness(img_raw, amax=2, amin=0.5):
#     img_raw = Image.open(img_path)
    cn = random.randint(amin*10, amax*10)
    imgcn = ImageEnhance.Brightness(img_raw).enhance(cn/10)
    return imgcn, 'b'+str(cn)


# In[ ]:


def crop(img_raw, xml_path, amax=50, amin=0):
#     img_raw = Image.open(img_path)
    crn1 = random.randint(amin, amax)
    crn2 = random.randint(amin, amax)
    crn3 = random.randint(amin, amax)
    crn4 = random.randint(amin, amax)
#     print(img_raw.size)
    imgsize = img_raw.size
    imgcr = img_raw.crop((crn1, crn2, img_raw.size[0]-crn3, img_raw.size[1]-crn4))
#     print(imgcr.size)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.iter('filename')
    
    xminl = []
    yminl = []
    xmaxl = []
    ymaxl = []
    for box in root.iter('bndbox'):
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)
    
        if int(xmin)-crn1 < 0:
            xmin = 0
        else:
            xmin = int(xmin)-crn1
        
        if int(ymin)-crn2 < 0:
            ymin = 0
        else:
            ymin = int(ymin)-crn2
        
        if int(xmax) > img_raw.size[0]-crn3-crn1:
            xmax = img_raw.size[0]-crn3-crn1-1
        else:
            xmax = int(xmax)
        
        if int(ymax) > img_raw.size[1]-crn4-crn2:
            ymax = img_raw.size[1]-crn4-crn2-1
        else:
            ymax = int(ymax)
            
        xminl.append(xmin)
        yminl.append(ymin)
        xmaxl.append(xmax)
        ymaxl.append(ymax)
#         print(xminl, yminl, xmaxl, ymaxl)
    
    return imgcr, xminl, yminl, xmaxl, ymaxl, 's' + str(int(imgcr.size[0])) + str(int(imgcr.size[1]))


# In[ ]:


def mode(mode, imgpath, xmlpath, amax, amin):
    if mode == 'h':
        img_raw = Image.open(imgpath)
        if amax == None and amax == None:
            return hue(img_raw)
        else:
            return hue(img_raw, amax, amin)
    if mode == 'c':
        img_raw = Image.open(imgpath)
        if amax == None and amax == None:
            return contrast(img_raw)
        else:
            return contrast(img_raw, amax, amin)
    if mode == 'b':
        img_raw = Image.open(imgpath)
        if amax == None and amax == None:
            return brightness(img_raw)
        else:
            return brightness(img_raw, amax, amin)
#     if mode == 'r':
#         img = cv2.imread(imgpath)
#         return rotate_image(img, random.randint(1, 24)*15)
    if mode == 's':
        img_raw = Image.open(imgpath)
        if amax == None and amax == None:
            return crop(img_raw, xmlpath)
        else:
            return crop(img_raw, xmlpath, amax, amin)
    if mode == 'a':
        if amax == None and amax == None:
            return angel(imgpath, xmlpath)
        else:
            return angel(imgpath, xmlpath, amax, amin)
    if mode == 'random':
        num = random.randint(1,6)
        img_raw = Image.open(imgpath)
        if num == 1:
            img_raw = Image.open(imgpath)
            img_new = crop(img_raw, xmlpath)
            imgnm = img_new[5]
            img_new[0].save('tmp.jpg')

            tree = ET.parse(xmlpath)
            root = tree.getroot()
            root.iter('filename')
            bn = 0
            for box in root.iter('bndbox'):
                box.find('xmin').text = str(img_new[1][bn])
                box.find('ymin').text = str(img_new[2][bn])
                box.find('xmax').text = str(img_new[3][bn])
                box.find('ymax').text = str(img_new[4][bn])
                bn = bn + 1
            for img_size in root.iter('size'):
                img_size.find('width').text = str(int(img_new[0].size[0]))
                img_size.find('height').text = str(int(img_new[0].size[1]))

            tree.write('tmp.xml')
 
            ans = angel('tmp.jpg', 'tmp.xml')
            img = cv2.cvtColor(ans[0][0], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img.astype('uint8'), 'RGB')
            i1 = brightness(image)
            i2 = contrast(i1[0])
            img_new = hue(i2[0])
            
            os.remove('tmp.jpg')
            os.remove('tmp.xml')

        if num == 2:
            img_raw = Image.open(imgpath)
            img_new = crop(img_raw, xmlpath)
            imgnm = img_new[5]
            img_new[0].save('tmp.jpg')

            tree = ET.parse(xmlpath)
            root = tree.getroot()
            root.iter('filename')
            bn = 0
            for box in root.iter('bndbox'):
                box.find('xmin').text = str(img_new[1][bn])
                box.find('ymin').text = str(img_new[2][bn])
                box.find('xmax').text = str(img_new[3][bn])
                box.find('ymax').text = str(img_new[4][bn])
                bn = bn + 1
            for img_size in root.iter('size'):
                img_size.find('width').text = str(int(img_new[0].size[0]))
                img_size.find('height').text = str(int(img_new[0].size[1]))

            tree.write('tmp.xml')
 
            ans = angel('tmp.jpg', 'tmp.xml')
            img = cv2.cvtColor(ans[0][0], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img.astype('uint8'), 'RGB')
            i1 = contrast(image)
            i2 = brightness(i1[0])
            img_new = hue(i2[0])
            os.remove('tmp.jpg')
            os.remove('tmp.xml')
        if num == 3:
            img_raw = Image.open(imgpath)
            img_new = crop(img_raw, xmlpath)
            imgnm = img_new[5]
            img_new[0].save('tmp.jpg')

            tree = ET.parse(xmlpath)
            root = tree.getroot()
            root.iter('filename')
            bn = 0
            for box in root.iter('bndbox'):
                box.find('xmin').text = str(img_new[1][bn])
                box.find('ymin').text = str(img_new[2][bn])
                box.find('xmax').text = str(img_new[3][bn])
                box.find('ymax').text = str(img_new[4][bn])
                bn = bn + 1
            for img_size in root.iter('size'):
                img_size.find('width').text = str(int(img_new[0].size[0]))
                img_size.find('height').text = str(int(img_new[0].size[1]))

            tree.write('tmp.xml')
 
            ans = angel('tmp.jpg', 'tmp.xml')
            img = cv2.cvtColor(ans[0][0], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img.astype('uint8'), 'RGB')
            i1 = brightness(image)
            i2 = hue(i1[0])
            img_new = contrast(i2[0])
            os.remove('tmp.jpg')
            os.remove('tmp.xml')
        if num == 4:
            img_raw = Image.open(imgpath)
            img_new = crop(img_raw, xmlpath)
            imgnm = img_new[5]
            img_new[0].save('tmp.jpg')

            tree = ET.parse(xmlpath)
            root = tree.getroot()
            root.iter('filename')
            bn = 0
            for box in root.iter('bndbox'):
                box.find('xmin').text = str(img_new[1][bn])
                box.find('ymin').text = str(img_new[2][bn])
                box.find('xmax').text = str(img_new[3][bn])
                box.find('ymax').text = str(img_new[4][bn])
                bn = bn + 1
            for img_size in root.iter('size'):
                img_size.find('width').text = str(int(img_new[0].size[0]))
                img_size.find('height').text = str(int(img_new[0].size[1]))

            tree.write('tmp.xml')
 
            ans = angel('tmp.jpg', 'tmp.xml')
            img = cv2.cvtColor(ans[0][0], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img.astype('uint8'), 'RGB')
            i1 = hue(image)
            i2 = brightness(i1[0])
            img_new = contrast(i2[0])
            os.remove('tmp.jpg')
            os.remove('tmp.xml')
        if num == 5:
            img_raw = Image.open(imgpath)
            img_new = crop(img_raw, xmlpath)
            imgnm = img_new[5]
            img_new[0].save('tmp.jpg')

            tree = ET.parse(xmlpath)
            root = tree.getroot()
            root.iter('filename')
            bn = 0
            for box in root.iter('bndbox'):
                box.find('xmin').text = str(img_new[1][bn])
                box.find('ymin').text = str(img_new[2][bn])
                box.find('xmax').text = str(img_new[3][bn])
                box.find('ymax').text = str(img_new[4][bn])
                bn = bn + 1
            for img_size in root.iter('size'):
                img_size.find('width').text = str(int(img_new[0].size[0]))
                img_size.find('height').text = str(int(img_new[0].size[1]))

            tree.write('tmp.xml')
 
            ans = angel('tmp.jpg', 'tmp.xml')
            img = cv2.cvtColor(ans[0][0], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img.astype('uint8'), 'RGB')
            i1 = contrast(image)
            i2 = hue(i1[0])
            img_new = brightness(i2[0])
            os.remove('tmp.jpg')
            os.remove('tmp.xml')
        if num == 6:
            img_raw = Image.open(imgpath)
            img_new = crop(img_raw, xmlpath)
            imgnm = img_new[5]
            img_new[0].save('tmp.jpg')

            tree = ET.parse(xmlpath)
            root = tree.getroot()
            root.iter('filename')
            bn = 0
            for box in root.iter('bndbox'):
                box.find('xmin').text = str(img_new[1][bn])
                box.find('ymin').text = str(img_new[2][bn])
                box.find('xmax').text = str(img_new[3][bn])
                box.find('ymax').text = str(img_new[4][bn])
                bn = bn + 1
            for img_size in root.iter('size'):
                img_size.find('width').text = str(int(img_new[0].size[0]))
                img_size.find('height').text = str(int(img_new[0].size[1]))

            tree.write('tmp.xml')
 
            ans = angel('tmp.jpg', 'tmp.xml')
            img = cv2.cvtColor(ans[0][0], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img.astype('uint8'), 'RGB')
            i1 = hue(image)
            i2 = contrast(i1[0])
            img_new = brightness(i2[0])
            os.remove('tmp.jpg')
            os.remove('tmp.xml')
        return img_new[0], ans[1], ans[2], ans[3], ans[4], i1[1]+i2[1]+img_new[1]+ans[7]+imgnm
        


# In[ ]:


if __name__ == "__main__":
    # Path of the images
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", action='store', dest='mode')
    parser.add_argument("--limit", type=int, dest='limit')
    parser.add_argument("--xmlpath", action='store', dest='xmlpath')
    parser.add_argument("--imgpath", action='store', dest='imgpath')
    parser.add_argument("--output_xmlpath", action='store', dest='rotated_xmlpath', default='./output_xml/')
    parser.add_argument("--output_imgpath", action='store', dest='rotated_imgpath', default='./output_img/')
    parser.add_argument("--max", type=float, dest='max', nargs='?')
    parser.add_argument("--min", type=float, dest='min', nargs='?')
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode")
#     parser.add_argument("limit")
#     parser.add_argument("xmlpath")
#     parser.add_argument("imgpath")
#     parser.add_argument("rotated_xmlpath")
#     parser.add_argument("rotated_imgpath")
    
    args = parser.parse_args()
    
    for l in range(int(args.limit)):
        for i in os.listdir(args.imgpath):
            print(l+1, i, end=' ')
            a, b = os.path.splitext(i)             
    #     angel(args.xmlpath, args.imgpath, args.rotated_xmlpath, args.rotated_imgpath)
#             if args.max == None and args.max == None:
#                 img_new = mode(args.mode, args.imgpath + a + '.jpg', args.xmlpath + a + '.xml')
#             else:
            img_new = mode(args.mode, args.imgpath + a + '.jpg', args.xmlpath + a + '.xml', args.max, args.min)

            if args.mode == 'h':
                img_new[0].save(args.rotated_imgpath + a + '_' + img_new[1] + '.jpg')
                tree = ET.parse(args.xmlpath + a + '.xml')
                tree.write(args.rotated_xmlpath + a + '_' + img_new[1] + '.xml')
                print(a + '_' + img_new[1])
            if args.mode == 'c':
                img_new[0].save(args.rotated_imgpath + a + '_' + img_new[1] + '.jpg')
                tree = ET.parse(args.xmlpath + a + '.xml')
                tree.write(args.rotated_xmlpath + a + '_' + img_new[1] + '.xml')
                print(a + '_' + img_new[1])
            if args.mode == 'b':
                img_new[0].save(args.rotated_imgpath + a + '_' + img_new[1] + '.jpg')
                tree = ET.parse(args.xmlpath + a + '.xml')
                tree.write(args.rotated_xmlpath + a + '_' + img_new[1] + '.xml')
                print(a + '_' + img_new[1])
            if args.mode == 's':
                img_new[0].save(args.rotated_imgpath + a + '_' + 's' + str(int(img_new[0].size[0])) + str(int(img_new[0].size[1])) + '.jpg')
                print(a + '_' + 's' + str(int(img_new[0].size[0])) + str(int(img_new[0].size[1])))
                tree = ET.parse(args.xmlpath + a + '.xml')
                root = tree.getroot()
                root.iter('filename')
                bn = 0
                for box in root.iter('bndbox'):
                    box.find('xmin').text = str(img_new[1][bn])
                    box.find('ymin').text = str(img_new[2][bn])
                    box.find('xmax').text = str(img_new[3][bn])
                    box.find('ymax').text = str(img_new[4][bn])
                    bn = bn + 1
                for img_size in root.iter('size'):
                    img_size.find('width').text = str(int(img_new[0].size[0]))
                    img_size.find('height').text = str(int(img_new[0].size[1]))

                tree.write(args.rotated_xmlpath + a + '_'+ 's' + str(int(img_new[0].size[0])) + str(int(img_new[0].size[1])) +'.xml')

            if args.mode == 'a':
                img = cv2.cvtColor(img_new[0][0], cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img.astype('uint8'), 'RGB')
                image.save(args.rotated_imgpath + a + '_' + img_new[7] + '.jpg')
                print(a + '_' + img_new[7])
                tree = ET.parse(args.xmlpath + a + '.xml')
                root = tree.getroot()
                root.iter('filename')
                bn = 0
                for box in root.iter('bndbox'):
                    box.find('xmin').text = str(img_new[1][bn])
                    box.find('ymin').text = str(img_new[2][bn])
                    box.find('xmax').text = str(img_new[3][bn])
                    box.find('ymax').text = str(img_new[4][bn])
                    bn = bn + 1
                for img_size in root.iter('size'):
                    img_size.find('width').text = str(img_new[5])
                    img_size.find('height').text = str(img_new[6])
                tree.write(args.rotated_xmlpath + a + '_' + img_new[7] + '.xml')

            if args.mode == 'random':
                img_new[0].save(args.rotated_imgpath + a + '_' + img_new[5] + '.jpg')
                print(a + '_' + img_new[5])
                tree = ET.parse(args.xmlpath + a + '.xml')
                root = tree.getroot()
                root.iter('filename')
                bn = 0
                for box in root.iter('bndbox'):
                    box.find('xmin').text = str(img_new[1][bn])
                    box.find('ymin').text = str(img_new[2][bn])
                    box.find('xmax').text = str(img_new[3][bn])
                    box.find('ymax').text = str(img_new[4][bn])
                    bn = bn + 1
                for img_size in root.iter('size'):
                    img_size.find('width').text = str(int(img_new[0].size[0]))
                    img_size.find('height').text = str(int(img_new[0].size[1]))

                tree.write(args.rotated_xmlpath + a + '_' + img_new[5] + '.xml')
                
                
            

