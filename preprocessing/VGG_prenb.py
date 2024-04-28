
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from retinaface import RetinaFace
import cv2
import os


print(torch.cuda.is_available())
device = torch.device('cuda')

class CustomVGG(nn.Module):
    def __init__(self):
        super(CustomVGG, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='casia-webface').to(device)
        # Adjust the output size of the linear layer
        self.fc = nn.Linear(in_features=512, out_features=1024*14*14)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        # Reshape the output tensor to (batch_size, 1024, 14, 14)
        x = x.view(x.size(0), 1024, 14, 14)
        return x


def preprocess_image(img, scale):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = RetinaFace.detect_faces(img)
    #print('passed check1')
    if len(faces) > 0:
        face_info = faces['face_1']
        facial_area = face_info['facial_area']
        left, top, right, bottom = facial_area

        margin = int(0.05 * (bottom - top))
        cropped_face = img[max(0, top - margin):bottom + margin, max(0, left - margin):right + margin]
        #print('passed check 2')
        size = (224, 224) if scale == 1 else (448, 448)
        normalized_img = transforms.ToTensor()(cv2.resize(cropped_face, size)).to(device)
        return normalized_img
    else:
        return None


def save_features(features, image_paths):
    for i, feature in enumerate(features):
        image_name = os.path.basename(image_paths[i])
        feature_path = f"data/CASIA-WebFace/features_scale_1/bonafide/raw/{image_name.replace('.jpg', '.pt')}"
        if not os.path.exists(feature_path):
            torch.save(feature, feature_path)
        pt_tensor = torch.load(feature_path)
        #print(f"Size 1: {image_name}.pt {pt_tensor.shape}")

def extract_patches(image_tensor):
    kernel_size, stride = 224, 224
    patches = image_tensor.unfold(2, kernel_size, stride).unfold(1, kernel_size, stride)
    patches = patches.contiguous().view(-1, 3, kernel_size, kernel_size)
    return patches

model = CustomVGG().to(device)
#print('model loaded')

image_dir = "data/CASIA-WebFace/images/bonafide/raw/"
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith((".jpg", ".jpeg", ".png"))]

batch_size = 3
batches_scale_1 = []
batch_scale_1 = []



for img_path in image_paths[5000:]:
    img = cv2.imread(img_path)
    image_name = os.path.basename(img_path)
    preprocessed_img_scale_1 = preprocess_image(img, 1)
    #preprocessed_img_scale_2 = preprocess_image(img, 2)

    if preprocessed_img_scale_1 is not None:
        batch_scale_1.append(preprocessed_img_scale_1)

    '''if preprocessed_img_scale_2 is not None:
        feature_path = f"data/CASIA-WebFace/features_scale_2/bonafide/raw/{image_name.replace('.jpg', '.pt')}"
        if not os.path.exists(feature_path):
            patches = extract_patches(preprocessed_img_scale_2)
            patches = (patches,)
            patches = torch.stack(patches,dim=0).squeeze(0)
            feature_scale_2 = model(patches)
            torch.save(feature_scale_2,feature_path)
        pt_tensor = torch.load(feature_path)
        print(f"Size 2:{image_name}.pt {pt_tensor.shape}")'''

    if len(batch_scale_1) == batch_size:
        batches_scale_1.append(batch_scale_1)
        #print('added to batch')
        batch_scale_1 = []


if len(batch_scale_1) > 0:
    batches_scale_1.append(batch_scale_1)

print('reached here')

start_index = 0
for batch_scale_1 in batches_scale_1:
    end_index = start_index + len(batch_scale_1)
    batch_tensor_scale_1 = torch.stack(batch_scale_1, dim=0)
    features_scale_1 = model(batch_tensor_scale_1)
    # re check this condition
    save_features(features_scale_1, image_paths[start_index:end_index])
    start_index = end_index


