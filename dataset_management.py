import os
import csv
import cv2

def generate_dataset(img, user_name, img_id, dataset_folder="data/users/"):
    os.makedirs(dataset_folder, exist_ok=True)

    # Save image with a clear naming convention
    img_filename = f"{img_id}_{user_name}.jpg"
    img_path = os.path.join(dataset_folder, img_filename)
    cv2.imwrite(img_path, img)


    # Append entry to CSV mapping
    csv_file = os.path.join(dataset_folder, "users.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f :
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["img_id","user_name", "img_path"]) # header
        writer.writerow([img_id, user_name, img_filename])
