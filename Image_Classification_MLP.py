import os
import cv2
import numpy as np
from skimage import feature
from skimage.morphology import skeletonize
from skimage.transform import resize
from sklearn.neural_network import MLPClassifier


def extract_feature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    features_list = []

    # invert image colors
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            # if partially white
            if col > 240:
                # make it dark black
                img[i][j] = 0
            # if partially black
            else:
                # make it completely white
                img[i][j] = 255

    # Skeletonize
    skeleton = skeletonize(img, method='lee')

    # Resize - Reshape: N x M -> N x N image
    resized_skeleton = resize(skeleton, (200, 200), order=1, anti_aliasing=False)

    # HoG Feature Detection
    hog_features, output = feature.hog(resized_skeleton, orientations=9, pixels_per_cell=(17, 17),
                                               cells_per_block=(2, 2), visualize=True, channel_axis=None)
    # Add HoG features
    features_list.append(hog_features)

    return hog_features


# dataset paths and total classes
training_path = './dataset/train/'
validation_path = './dataset/val/'
classes = os.listdir(training_path)


def process_images(path, classes):
    images = []
    labels = []
    count = 0

    for class_name in classes:
        dir_path = path + str(class_name)
        filenames = [img_name for img_name in os.listdir(dir_path)]
        print("len files: ", len(filenames))

        for filename in filenames:
            image_path = dir_path + '/' + filename
            # OpenCV can't resize broken images
            try:
                img = extract_feature(image_path)
            # if broken image, just bypass
            except Exception as e:
                print(str(e))
                continue
            count += 1
            print("Image Processed: ", count)
            img = np.array(img)
            images.append(img)
            labels.append(class_name)

    return images, labels


# Process Training Data
X_train, y_train = process_images(training_path, classes)

# Train Model
cls_model = MLPClassifier(hidden_layer_sizes=(50), activation='logistic', random_state=10)
print("----------------- Model Training Started -----------------")
cls_model.fit(X_train, y_train)
print("----------------- Model Training Ended -----------------")

# Process Validation Data
print("----------------- Validation Data Processing -----------------")
validation_data, original_classes = process_images(validation_path, classes)
print("----------------- Processing Complete -----------------")

# Predict Validation Data
predicted_classes = cls_model.predict(validation_data)

# Find Classification Model Accuracy
model_accuracy = None

if len(original_classes) == len(predicted_classes):
    true_predictions = 0
    false_predictions = 0

    for idx, org_class in enumerate(original_classes):
        if org_class == predicted_classes[idx]:
            true_predictions += 1
        else:
            false_predictions += 1

    model_accuracy = (true_predictions / (true_predictions + false_predictions)) * 100
    model_accuracy = round(model_accuracy, 2)

print("MODEL ACCURACY: ", model_accuracy)
