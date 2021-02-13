import os
import glob

path = input("Enter path: ")
os.makedirs(path, exist_ok=True)
num = 0

for name in glob.glob(os.path.join(path, '*.jpg')):
    fileName = name.split('\\')[1]
    number = int(fileName.split('.')[0])
    if number > num:
        num = number

num += 1

while True:
    os.system(f'ffmpeg -i http://wpilibpi.local:1183/stream.mjpg -y -vframes 1 {path}\\{num}.jpg')
    num = num + 1
    input("Press enter to take new picture")