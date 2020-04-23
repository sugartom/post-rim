import torchvision
from torchvision.models.detection import FasterRCNN
from PIL import Image
import torchvision.transforms as transforms
import time

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

img_path = '/home/yitao/Documents/fun-project/tensorflow-related/post-rim/test/test_ssd/dog.jpg'
img = Image.open(img_path)
img = img.resize((800, 800))
transform = transforms.Compose([transforms.ToTensor()])
img = transform(img)

run_time = 10
batch_size = 4
images = []
for i in range(batch_size):
  images.append(img)

for i in range(10):
  start = time.time()
  predictions = model(images)
  end = time.time()
  print("duration = %f" % (end - start))

# print(predictions)