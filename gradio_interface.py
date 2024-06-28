#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr

DATA_DIR = "/app/data/forest_fire"
TRAIN_DIR = os.path.join(DATA_DIR, "Training and Validation")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

transform = transforms.Compose(
    [
        transforms.ToPILImage(), transforms.Resize(
            (224, 224)), transforms.ToTensor(), transforms.Normalize(
                mean=[
                    0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225]), ])


def load_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 2)
    model.load_state_dict(
        torch.load("/app/model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model


def predict(image):
    model = load_model()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        if predicted.item() == 0:
            return "Pozar"
        else:
            return "Brak pozaru"


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Textbox(label="Predykcje"),
    title="Wykrywanie pozaru",
    description="Przeslij zdjecie by zidentyfikowac pozar",
)

if __name__ == "__main__":
    interface.launch(share=True)
