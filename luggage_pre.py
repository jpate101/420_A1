import os
import cv2
import os.path
import csv
import glob


def mkdir(path):

    import os

 
    path = path.strip()
 
    path = path.rstrip("\\")


    isExists = os.path.exists(path)


    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


mkpath = "luggage_data\\train\\0"
mkdir(mkpath)
mkpath = "luggage_data\\train\\1"
mkdir(mkpath)
mkpath = "luggage_data\\val\\0"
mkdir(mkpath)
mkpath = "luggage_data\\val\\1"
mkdir(mkpath)

names = []
luggage = []
with open('Train_Data\Train.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        names.append(row[0])
        luggage.append(row[13])
dict = {}

for j in range(len(names)):
    dict[names[j]] = luggage[j]

k = 0
for pngfile in glob.glob('Train_Data\Parsed\*.png'):
    img = cv2.imread(pngfile)
    if dict[os.path.basename(pngfile)] == '0':
        imgpath = os.path.join(
            'luggage_data\\train\\0', os.path.basename(pngfile))
        cv2.imwrite(imgpath, img)
    elif dict[os.path.basename(pngfile)] == '1':
        imgpath = os.path.join(
            'luggage_data\\train\\1', os.path.basename(pngfile))
        cv2.imwrite(imgpath, img)
    # elif dict[os.path.basename(pngfile)] == '-1':
    #     imgpath = os.path.join(
    #         'Train_Data\luggage_data\\train\\-1', os.path.basename(pngfile))

    k = k+1

print(k)
# exit()

names = []
luggage = []
with open('Test_Data\Test.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        names.append(row[0])
        luggage.append(row[13])
dict = {}

for j in range(len(names)):
    dict[names[j]] = luggage[j]

k = 0
for pngfile in glob.glob('Test_Data\Originals\*.png'):
    img = cv2.imread(pngfile)
    if dict[os.path.basename(pngfile)] == '0':
        imgpath = os.path.join(
            'luggage_data\\val\\0', os.path.basename(pngfile))
        cv2.imwrite(imgpath, img)
    elif dict[os.path.basename(pngfile)] == '1':
        imgpath = os.path.join(
            'luggage_data\\val\\1', os.path.basename(pngfile))
        cv2.imwrite(imgpath, img)
    # elif dict[os.path.basename(pngfile)] == '-1':
    #     imgpath = os.path.join(
    #         'luggage_data\\val\\-1', os.path.basename(pngfile))

    k = k+1

print(k)
exit()
