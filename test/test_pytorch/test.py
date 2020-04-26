import numpy as np

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from PIL import Image
import torchvision.transforms as transforms
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

model.eval()

img_path = '/home/yitao/Documents/fun-project/tensorflow-related/post-rim/test/test_ssd/dog.jpg'
img = Image.open(img_path)
img = img.resize((800, 800))
transform = transforms.Compose([transforms.ToTensor()])
img = transform(img).to(device)

run_time = 5
batch_array = [1, 2, 3, 4]

for batch_size in batch_array:
  images = []
  for i in range(batch_size):
    images.append(img)

  for i in range(run_time):
    start = time.time()
    predictions = model(images)
    end = time.time()
    print("duration = %f for batch size of %d" % (end - start, batch_size))


