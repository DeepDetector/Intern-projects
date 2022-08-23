import torch
import os
import cv2 as cv
from PIL import Image
from models.resnet import resnet50
from torchvision import transforms

data_transform = {
    "val": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])}

if __name__ == '__main__':
    dirpath = './data/0to1/'
    imgs = os.listdir(dirpath)
    net = resnet50(num_classes=4)
    net.load_state_dict(torch.load('./weights/resnet_epoch100_256.pkl'))
    net = net.to('cuda')
    net.eval()
    with torch.no_grad():
        for imgname in imgs:
            img = cv.imread(dirpath+imgname)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img = data_transform['val'](img).unsqueeze(0)
            img = img.to('cuda')
            outputs = net(img)
            label = torch.softmax(outputs, dim=1)
            predict = torch.max(label, dim=1)[1]
            score = torch.max(label, dim=1)[0]
            print(outputs)
            print(predict[0].cpu().numpy())
