from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd
import os


### Note: To test the model, you are expected to possess a directory with enough
### test data images, equally processed as the training data!
# -> Adjust preprocessing script to generate a random test sample from the total sample
# -> Adjust the create_target script to retrieve the target labels for the test data
# -> Then: Define the correct local file paths for 1) the model, 2) the test data ("img_test_dir_path")


model = load_model('./model')
img_test_dir_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_test_proc/images"
labels = pd.read_csv("/home/sebastian/code/ipl1988/test_data_labels.csv")['Label'].values

## Sort labels alphanumerical so that the image loading function can treat them optimally##
images = os.listdir(img_test_dir_path + "/images")
# Zipping images and labels together
zipped = list(zip(images, labels))
# Sorting the zipped list by the image names (alphanumeric order)
sorted_zipped = sorted(zipped, key=lambda x: x[0])
# Unzipping the sorted list back into images and labels
sorted_images, sorted_labels = zip(*sorted_zipped)
# Converting back to labels list
sorted_labels = list(sorted_labels)

test_dataset = image_dataset_from_directory(
    directory=img_test_dir_path,  # Path to test images
    labels=sorted_labels,  # Labels for the test data
    label_mode='int',
    color_mode='grayscale',
    batch_size=32,
    image_size=(150, 150),
    shuffle=True
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
