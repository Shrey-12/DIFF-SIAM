
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from retinaface import RetinaFace
import cv2
import os

# Load the Wide ResNet 50 model, ensuring correct model name
model = models.wide_resnet50_2(pretrained=True)
count = 0

image_dir = "data/CASIA-WebFace/images/bonafide/raw/"
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith((".jpg", ".jpeg", ".png"))]

def load_and_preprocess(img_path, scale):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Assuming RetinaFace is used for face detection
    faces = RetinaFace.detect_faces(img)
    if len(faces) > 0:
        # Assuming you want to use the first detected face
        face_info = faces['face_1']

        # Extract the bounding box information
        score = face_info['score']
        facial_area = face_info['facial_area']
        left, top, right, bottom = facial_area

        # Crop the face with a margin of 5% of the detected bounding box height
        margin = int(0.05 * (bottom - top))
        cropped_face = img[max(0, top - margin):bottom + margin, max(0, left - margin):right + margin]

        # Resize the image to the desired scale
        size = (224, 224) if scale == 1 else (448, 448)
        normalized_img = transforms.ToTensor()(cv2.resize(cropped_face, size))

        return normalized_img
    else:
        print(f"No faces detected in {img_path}")
        count-=1
        return None

for img_path in image_paths:
    count+=1
    image_name = os.path.basename(img_path)
    for scale in [1, 2]:
        feature_path = f"../../../feature_scale_{scale}/{image_name.replace('.jpg', '.pt')}"
        if os.path.exists(feature_path):
            continue
        normalized_img = load_and_preprocess(img_path, scale)

        if normalized_img is not None:
            features = model(normalized_img.unsqueeze(0)).squeeze(0)
            
            # model, dir
            torch.save(features, feature_path)

