#coding=utf-8
import os
import os.path
import shutil  #Python文件复制相应模块
 
def GetFileNameAndExt(filename):
    (filepath,tempfilename) = os.path.split(filename);
    (shotname,extension) = os.path.splitext(tempfilename);
    return shotname,extension
 
source_dir='/home/king/test/python/train/data/all/train'
# label_dir='/home/king/test/python/train/data/all/new_dog-cat'
annotion_Dog_dir='/home/king/test/python/train/data/all/new_dog'
annotion_Cat_dir='/home/king/test/python/train/data/all/new_cat'
 
##1.将指定A目录下的文件名取出,并将文件名文本和文件后缀拆分出来
img=os.listdir(source_dir)  #得到文件夹下所有文件名称
s=[]
for fileNum in img: #遍历文件夹
    if not os.path.isdir(fileNum): #判断是否是文件夹,不是文件夹才打开
        # print(fileNum)  #打印出文件名
        imgname= os.path.join(source_dir,fileNum)
        # print(imgname)  #打印出文件路径
        (imgpath,tempimgname) = os.path.split(imgname); #将路径与文件名分开
        # print(imgpath)
        if 'dog' in tempimgname:
            tempimgname = tempimgname.split('.')[1]
            # print(tempimgname)
            new_name = tempimgname + '.jpg'
            print(new_name)
            new_path = os.path.join(annotion_Dog_dir, new_name)
            # print('dog\n', new_path)
            shutil.copy(imgname,annotion_Dog_dir)
        else:
            tempimgname = tempimgname.split('.')[1]
            # print(tempimgname)
            new_name = tempimgname + '.jpg'
            print(new_name)
            new_path = os.path.join(annotion_Cat_dir, new_name)
            # print('cat\n', new_path)
            shutil.copy(imgname,annotion_Cat_dir)
        # (shotname,extension) = os.path.splitext(tempimgname); #将文件名文本与文件后缀分开
    # print(shotname)
    # print(extension)
    print('~~~~')
##2.将取出来的文件名文本与特定后缀拼接,再与路径B拼接,得到B目录下的文件	
##3.根据得到的xml文件名,将对应文件拷贝到指定目录C
    # print(new_path)

	
