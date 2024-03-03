from torch import load

path = "../data/CASIA-WebFace/features_scale_1/bonafide/raw/0000512_031.pt"
try:
    loaded_data = load(path)
    print("File loaded successfully.")
except Exception as e:
    print(f"Error loading file: {str(e)}")

