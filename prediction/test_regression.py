import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
from prediction import train_regression_model


def predict(image_path, model_path):
    transform = A.Compose(
        [A.Resize(224, 224), A.Normalize(mean=(0.2611, 0.4760, 0.3845), std=(0.2331, 0.2544, 0.2608)),
         ToTensorV2()])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)["image"]
    image = image.unsqueeze(0)

    model = train_regression_model.ResNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()

    with torch.no_grad():
        pred = model.forward(image)  # torch.Size([1])

    return pred.item()
