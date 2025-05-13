import tensorflow as tf
import matplotlib.pyplot as plt
import time

from clearml import Task, Dataset
from keras import layers, Model

task = Task.init(project_name="PokemonClassification", task_name="AugmentationTask", output_uri=True)

# set lerning on GPU/CPU
useCPU = True  # 'CPU' or 'GPU'

if (useCPU == True):
    tf.config.set_visible_devices([], 'GPU')  # hide GPU
    print("Aviable devides:", tf.config.get_visible_devices())
else:
    gpu_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpu_devices, 'GPU') # use GPU
    print("Aviable devides:", tf.config.get_visible_devices())


# Load the dataset from clearml
#dataPath = "dataFixed"
dataPath = Dataset.get(dataset_id="13db2337377344489645212c8c30ca17").get_local_copy()

# Set the parameters
params = {'batch_size': 16,# liczba obrazow na raz
          'img_height': 128,# rozmiar obrazu po skalowaniu
          'img_width': 128,
          'epochs': 15}
task.connect(params)

# split the data
train_ds = tf.keras.utils.image_dataset_from_directory(
  dataPath,
  validation_split=0.2,
  subset='training',
  seed=123,
  image_size=(params['img_height'], params['img_width']),
  batch_size=params['batch_size'])

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataPath,
  validation_split=0.2,
  subset='validation',
  seed=123,
  image_size=(params['img_height'], params['img_width']),
  batch_size=params['batch_size'])


# Get the class names
class_names = train_ds.class_names
class_count = len(class_names)
print(class_names)

# Normalization layer from 0,255 to 0,1
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
val_ds = val_ds.map(lambda x, y: (data_augmentation(x), y))


# prepering model
base_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=(params['img_height'], params['img_width'], 3),
            classes=class_count,
        )
x = layers.GlobalAveragePooling2D()(base_model.output)

# Add a fully connected layer with a softmax activation for multi-class classification
predictions = layers.Dense(class_count, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
model.summary()

# model lerning
start_time = time.time()
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=params["epochs"],
  batch_size=params["batch_size"]
)
training_time = time.time() - start_time


#log results to clearml
logger = task.get_logger()
logger.report_scalar("Overall Metrics", "Training Time (s)", training_time, iteration=0)

val_loss, val_accuracy = model.evaluate(val_ds)
logger.report_scalar("Overall Metrics", "Validation Accuracy %", val_accuracy*100, iteration=0)
logger.report_scalar("Overall Metrics", "Validation Loss", val_loss, iteration=0)

for epoch, (acc, val_acc, loss, val_loss) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'])):
    logger.report_scalar("Epoch Accuracy Metrics", "Training Accuracy %", acc*100, iteration=epoch + 1)
    logger.report_scalar("Epoch Accuracy Metrics", "Validation Accuracy %", val_acc*100, iteration=epoch + 1)
    logger.report_scalar("Epoch Loss Metrics", "Training Loss", loss, iteration=epoch + 1)
    logger.report_scalar("Epoch Loss Metrics", "Validation Loss", val_loss, iteration=epoch + 1)



# show results
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")

print(f"\nTotal training time: {training_time:.2f} seconds")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(params["epochs"])

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



