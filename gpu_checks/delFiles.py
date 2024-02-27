import os

def delete_unused_jpg_files(jpg_directory, pt_directory):
    jpg_files = [f for f in os.listdir(jpg_directory) if f.endswith('.jpg')]

    for jpg_file in jpg_files:
        jpg_path = os.path.join(jpg_directory, jpg_file)
        pt_path = os.path.join(pt_directory, jpg_file.replace('.jpg', '.pt'))

        if not os.path.exists(pt_path):
            print(f"Deleting {jpg_file}")
            os.remove(jpg_path)

if __name__ == "__main__":
    # Replace these paths with your actual directories
    pt_directory = "../data/CASIA-WebFace/features_scale_1/bonafide/raw"
    jpg_directory = "../data/CASIA-WebFace/images/bonafide/raw"
    try:
        delete_unused_jpg_files(jpg_directory, pt_directory)
    except:
        print("wrong directories")
