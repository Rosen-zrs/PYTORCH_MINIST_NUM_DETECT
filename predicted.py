import torch
import cv2 as cv
from main import Detector
from PIL import Image
from torchvision import transforms
import numpy

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

def predict(img_path):
    src = cv.imread('picture/_' + img_path)
    dst = cv.resize(src, (200, 200))
    cv.imshow("dst", dst)
    cv.waitKey(0)

    net = Detector()
    net.load_state_dict(torch.load('model.pkl'))
    net.eval()
    
    torch.no_grad()

    img = Image.open('picture/_' + img_path)
    img = img.resize((28, 28))
    img = img.convert('L')
    img = transform(img).unsqueeze(0)


    outputs = net(img)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.item()

    print('Result: ', predicted)

def pre_deal_pict(img_path):
    src = cv.imread(img_path)
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src = cv.GaussianBlur(src, (5, 5), 0)      #进行高斯模糊
    ret, dst = cv.threshold(src, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(dst, element)
    cv.imwrite('picture/_' + img_path, dst)



if __name__ == '__main__':
    img_path = "1.png"
    pre_deal_pict(img_path)
    predict(img_path)