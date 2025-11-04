from fastapi import FastAPI, HTTPException, UploadFile, File
from torchvision import transforms
# from pydantic import BaseModel
import streamlit as st
from PIL import Image
import torch.nn as nn
import uvicorn
import torch
import io



classes = {
    0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed',
    6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle', 10: 'bowl', 11: 'boy',
    12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can', 17: 'castle',
    18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 22: 'clock',
    23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'crab', 27: 'crocodile',
    28: 'cup', 29: 'dinosaur', 30: 'dolphin', 31: 'elephant', 32: 'flatfish',
    33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house',
    38: 'kangaroo', 39: 'keyboard', 40: 'lamp', 41: 'lawn_mower', 42: 'leopard',
    43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree',
    48: 'motorcycle', 49: 'mountain', 50: 'mouse', 51: 'mushroom', 52: 'oak_tree',
    53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear',
    58: 'pickup_truck', 59: 'pine_tree', 60: 'plain', 61: 'plate', 62: 'poppy',
    63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray',
    68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark',
    74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake',
    79: 'spider', 80: 'squirrel', 81: 'streetcar', 82: 'sunflower',
    83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television',
    88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle',
    94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'
}

class CheckImageAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),  # 32×32
            nn.MaxPool2d(2),  # 16×16

            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8×8

            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImageAlexNet()
model.load_state_dict(torch.load('check_flowers_model.pth', map_location=device))
model.to(device)
model.eval()

# app = FastAPI()

# @app.post('/predict')
# async def check_image(file:UploadFile = File(...)):
#     try:
#         data = await file.read()
#         if not data:
#             raise HTTPException(status_code=400, detail='File not Found')
#
#         img = Image.open(io.BytesIO(data))
#         img_tensor = transform(img).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             prediction = model(img_tensor)
#             result = prediction.argmax(dim=1).item()
#             return {f'class': classes[result]}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f'{e}')
#
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

st.title('Cifar-100 Model')
st.text('Загрузите изображение, и модель попробует её распознать.')

mnist_image = st.file_uploader('Выберите изображение', type=['PNG', 'JPG', 'JPEG', 'SVG', 'WEBP'])

if not mnist_image:
    st.info('Загрузите изображение')
else:
    st.image(mnist_image, caption='Загруженное изображение')

    if st.button('Распознать'):
        try:
            image = Image.open(mnist_image)
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                y_prediction = model(image_tensor)
                prediction = y_prediction.argmax(dim=1).item()
            st.success(f'Модель думает, что это: {classes[prediction]}')

        except Exception as e:
            st.error(f'Ошибка: {str(e)}')
