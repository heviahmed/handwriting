import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Dataset paths
dataset_path = r'C:\Users\pc\Documents\heviDataset\Images-20241223T085223Z-001\Images'
csv_path = r'C:\Users\pc\Documents\heviDataset\Labels-20241223T084707Z-001\Labels\lines-labels.csv'

# Image size and sequence length
IMG_WIDTH, IMG_HEIGHT = 224, 224  # ResNet expects 224x224 images
SEQ_LENGTH=10


# Load and preprocess dataset
def load_and_preprocess_dataset():
    # Define dataset path and CSV path
    dataset_path = r'C:\Users\pc\Documents\heviDataset\Images-20241223T085223Z-001\Images'
    csv_path = r'C:\Users\pc\Documents\heviDataset\Labels-20241223T084707Z-001\Labels\lines-labels.csv'

    # Read the folder names in the dataset_path
    folder_names = [folder_name.strip() for folder_name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder_name))]
    print(f"Total folders found: {len(folder_names)}")

    # Read the CSV file
    labels_df = pd.read_csv(csv_path)

    # Strip and normalize the 'ID' column for consistency
    labels_df['ID'] = labels_df['ID'].str.strip()
    print(f"Total IDs in CSV: {len(labels_df)}")

    # Filter valid folder IDs
    filtered_labels_df = labels_df[labels_df['ID'].isin(folder_names)].copy()

    # Debug: Find folders and IDs that don't match
    non_matching_folders = [folder for folder in folder_names if folder not in labels_df['ID'].values]
    non_matching_ids = [folder for folder in labels_df['ID'].values if folder not in folder_names]

    print(f"Folders not matching any IDs in the CSV: {len(non_matching_folders)}")
    print(f"IDs in the CSV not matching any folders: {len(non_matching_ids)}")

    # Count matching IDs
    matching_ids = labels_df[labels_df['ID'].isin(folder_names)]
    matching_count = len(matching_ids)

    print(f"Number of matching IDs: {matching_count}")

    label_encoder = LabelEncoder()
    filtered_labels_df['label'] = label_encoder.fit_transform(filtered_labels_df['sentence'])
    print(f"Unique labels after encoding: {filtered_labels_df['label'].unique()}")

    images = []
    sequences = []
    labels = []

    for sentence_id in os.listdir(dataset_path):
        sentence_folder = os.path.join(dataset_path, sentence_id)
        if os.path.isdir(sentence_folder) and sentence_id in filtered_labels_df['ID'].values:
            label = filtered_labels_df[filtered_labels_df['ID'] == sentence_id]['label'].values[0]
            sentence_images = []
            for img_file in sorted(os.listdir(sentence_folder))[:SEQ_LENGTH]:  # Ensure sequence length matches
                img_path = os.path.join(sentence_folder, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping unreadable image: {img_path}")
                    continue
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
                sentence_images.append(img)
            if len(sentence_images) == SEQ_LENGTH:  # Only include complete sequences
                sequences.append(np.array(sentence_images))
                labels.append(label)

    sequences = np.array(sequences)
    labels = np.array(labels)
    print(f"Loaded {len(sequences)} sequences and {len(labels)} labels.")
    return sequences, labels, label_encoder


# Load dataset
sequences, labels, label_encoder = load_and_preprocess_dataset()

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Convert labels to categorical////////////////////////////////num classes have been changed
num_classes = max(labels) + 1
print(f"Recomputed num_classes: {num_classes}")

#New codes/start
print(f"y_train labels: {y_train[:10]}")  # Show the first 10 labels for debugging
print(f"Unique labels in y_train: {np.unique(y_train)}")  # Show all unique labels
print(f"Maximum label in y_train: {max(y_train)}")  # Find the highest label value
print(f"Number of classes (num_classes): {num_classes}")  # Check num_classes value
#end

y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)



# Define ResNet + BiLSTM Model
def build_resnet_bilstm_model(input_shape, num_classes):
    # ResNet backbone
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in resnet_base.layers:
        layer.trainable = False  # Freeze ResNet layers

    # Feature extractor for each image in the sequence
    input_layer = Input(shape=input_shape)
    time_distributed = TimeDistributed(resnet_base)(input_layer)
    time_distributed = TimeDistributed(Flatten())(time_distributed)

    # BiLSTM for sequential modeling
    bilstm = Bidirectional(LSTM(256, return_sequences=False))(time_distributed)
    dropout = Dropout(0.5)(bilstm)

    # Fully connected output
    output_layer = Dense(num_classes, activation='softmax')(dropout)

    # Compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Build model
input_shape = (SEQ_LENGTH, IMG_WIDTH, IMG_HEIGHT, 3)  # Sequence of images
model = build_resnet_bilstm_model(input_shape, num_classes)

# Print model summary
model.summary()

# Add learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler]
)

# Save the trained model
model.save('resnet_bilstm_kurdish_handwriting.keras')
print("Model saved as 'resnet_bilstm_kurdish_handwriting.keras'")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')