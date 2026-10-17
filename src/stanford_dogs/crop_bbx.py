import os
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET

RAW_IMAGES = Path("data/raw/images/Images")
RAW_ANNOTS = Path("data/raw/annotations/Annotation")
PROCESSED_IMAGES = Path("data/processed/images/Images")

def crop_and_save_bbx():

    os.makedirs(PROCESSED_IMAGES, exist_ok= True)

    for breed_dir in RAW_IMAGES.iterdir():

        if not breed_dir.is_dir():
            continue    

        for img_path in breed_dir.glob("*.jpg"):
            annot = RAW_ANNOTS / breed_dir.name / img_path.stem
            full_image = cv2.imread(img_path)

            tree = ET.parse(annot)
            root = tree.getroot()

            objects = root.findall("object") # all dogs in the image
            for i, obj in enumerate(objects):
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                cropped_img = full_image[ymin:ymax, xmin:xmax]

                out_dir = PROCESSED_IMAGES / breed_dir.name
                os.makedirs(out_dir, exist_ok=True) 
                if len(objects) > 1:
                    filename = f"{img_path.stem}_{i}{img_path.suffix}" # different names case more than one dog
                else:
                    filename = img_path.name

                cropped_img_path = out_dir / filename
                cv2.imwrite(cropped_img_path, cropped_img)


if __name__ == '__main__':
    crop_and_save_bbx()