from abc import ABC, abstractmethod
from stanford_dogs.config import (
    Final_Model_Path,
    Test_Example_Path,
    IMG_SIZE,
    DOG_BREED_CLASSES,
    NUM_CLASSES,
)
import torch
import PIL.Image as Image
import logging
from stanford_dogs.transforms import get_transforms
from stanford_dogs.model import build_model
from stanford_dogs.utils import configure_logging

logger = logging.getLogger(__name__)


def _load_and_transform_img(img_source, transforms):
    if isinstance(img_source, Image.Image):
        img = img_source.convert('RGB')
    else:
        img = Image.open(img_source).convert('RGB')
    transformed_img = transforms(img)
    return transformed_img.unsqueeze(0)

def _load_model(model_path, device):
    logger.info("Loading checkpoint from %s onto %s", model_path, device)
    model = build_model(NUM_CLASSES)
    saved_checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(saved_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class ModelBase(ABC):
    @abstractmethod
    def predict(self,  img_source) -> str: pass

class DogClassifier(ModelBase):
    def __init__(self, model_path, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_model(model_path=model_path, device=self.device)
        _, self.transforms = get_transforms(img_size=IMG_SIZE)

    @torch.no_grad()
    def predict(self, img_source):
        transformed_img = _load_and_transform_img(img_source= img_source, transforms= self.transforms)
        transformed_img = transformed_img.to(self.device)
        output = self.model(transformed_img)
        result = torch.argmax(output, dim=1).item()
        return DOG_BREED_CLASSES[result]



if __name__ == '__main__':
    configure_logging()
    dog_classifier = DogClassifier(model_path=Final_Model_Path)
    logger.info('Predicted Breed: %s', dog_classifier.predict(img_source= Test_Example_Path))