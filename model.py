import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import os

class Predictor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
        
        model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        
        img_res = cv2.resize(image, (256, 256))
        img_res = img_res.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_res - mean) / std
        
        img_input = img_norm.transpose(2, 0, 1)
        input_tensor = torch.from_numpy(img_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = output.squeeze().cpu().numpy()
            
        mask_256 = (pred > 0.5).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        mask_256 = cv2.erode(mask_256, kernel, iterations=1)
        
        mask_final = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return (mask_final * 255).astype(np.uint8)

_predictor = Predictor()

def predict(image: np.ndarray) -> np.ndarray:
    return _predictor.predict(image)