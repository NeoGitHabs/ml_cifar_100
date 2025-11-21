# main.py — рабочая версия для Streamlit (CIFAR-100)

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import io

# Список из 100 классов CIFAR-100 (в правильном порядке!)
classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
    'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


# Точная копия твоей модели VGG16 для CIFAR-100
class VGG16_CIFAR(nn.Module):
    def __init__(self, num_classes=100, dropout=0.325):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Точный трансформ, как при обучении!
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4866, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    ),
])


# Загрузка модели
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16_CIFAR().to(device)
    model.load_state_dict(torch.load('cifar100_vgg.pth', map_location=device))
    model.eval()
    return model, device


model, device = load_model()

# Интерфейс Streamlit
st.title('CIFAR-100 Классификатор')
st.write('Модель VGG16, обучена 300 эпох → **70.24%** на тесте')

uploaded_file = st.file_uploader("Загрузи фото (32x32 подойдёт лучше всего)", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Твоё изображение", use_container_width=True)

    if st.button('Распознать'):
        with st.spinner('Думаю...'):
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                pred_idx = output.argmax(dim=1).item()
                confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

            st.success(f'**Это: {classes[pred_idx].upper()}**')
            st.write(f"Уверенность: {confidence:.1%}")
            st.balloons()


st.caption("Совет: картинки 32×32 пикселей дают самый точный результат")

# app = FastAPI(title="CIFAR-100 VGG16")
#
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(400, detail="Файл не изображение")
#
#     data = await file.read()
#     img = Image.open(io.BytesIO(data)).convert("RGB")
#
#     img_tensor = transform(img).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         output = model(img_tensor)
#         pred_idx = output.argmax(dim=1).item()
#
#     return {"class": classes[pred_idx]}
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
