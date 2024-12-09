import numpy as np
from tensorflow.keras.utils import to_categorical
from A.baseline_a import baseline_model_a
from B.baseline_b import baseline_model_b

# implementation of task A
# load breastmnist dataset
breast_data = np.load('Datasets/breastmnist.npz')

train_images = blood_data['train_images']
train_labels = blood_data['train_labels']
val_images = blood_data['val_images']
val_labels = blood_data['val_labels']
test_images = blood_data['test_images']
test_labels = blood_data['test_labels']

# data normalization
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# 
input_shape = train_images.shape[1:] + (1,)

# create baseline model of task a
baseline_model_a = baseline_model_a(input_shape)

# compile baseline model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# set early_stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_images[..., np.newaxis],
    train_labels,
    validation_data=(val_images[..., np.newaxis], val_labels),
    epochs=100,
    batch_size=8,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(test_images[..., np.newaxis], test_labels)
print(f"Test Loss of A: {test_loss:.4f}")
print(f"Test Accuracy of A: {test_accuracy:.4f}")

plt.plot(history.history['accuracy'], label='Train Accuracy ')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy of A')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss of A')
plt.show()



# implementation of task B
# load bloodmnist dataset
blood_data = np.load('Datasets/bloodmnist.npz')

train_images = blood_data['train_images']
train_labels = blood_data['train_labels']
val_images = blood_data['val_images']
val_labels = blood_data['val_labels']
test_images = blood_data['test_images']
test_labels = blood_data['test_labels']


train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels, num_classes=8)
val_labels = to_categorical(val_labels, num_classes=8)
test_labels = to_categorical(test_labels, num_classes=8)

input_shape = train_images.shape[1:]

baseline_model_b = baseline_model_b(input_shape)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_images[..., np.newaxis],  # 添加通道维度
    train_labels,
    validation_data=(val_images[..., np.newaxis], val_labels),
    epochs=100,
    batch_size=8,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(test_images[..., np.newaxis], test_labels)
print(f"Test Loss of A: {test_loss:.4f}")
print(f"Test Accuracy of A: {test_accuracy:.4f}")

plt.plot(history.history['accuracy'], label='Train Accuracy ')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy of A')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss of A')
plt.show()

