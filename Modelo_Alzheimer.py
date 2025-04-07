import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random
from tqdm import tqdm
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ========== CONFIG ========== #
base_path = r"C:\Users\dafne\OneDrive\Documentos\Escritorio\Alzheimer\Alzheimer_s Dataset"
train_dir = os.path.join(base_path, "train")
val_dir = os.path.join(base_path, "val")  # Se crea si no existe
test_dir = os.path.join(base_path, "test")
val_split = 0.2
seed = 42

# ========== DIVIDE TRAIN EN VALIDACIÓN ========== #
def create_validation_split(train_dir, val_dir, val_split=0.2):
    if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
        print("✔️ La carpeta de validación ya existe, no se moverán imágenes.")
        return

    print("📁 Creando conjunto de validación...")
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_train_path):
            continue

        images = os.listdir(class_train_path)
        val_size = int(len(images) * val_split)
        val_images = random.sample(images, val_size)

        class_val_path = os.path.join(val_dir, class_name)
        os.makedirs(class_val_path, exist_ok=True)

        for img in val_images:
            shutil.move(os.path.join(class_train_path, img),
                        os.path.join(class_val_path, img))

    print("✅ División completada.")

create_validation_split(train_dir, val_dir, val_split)

# ========== GENERADORES ========== #
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)

val_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ========== MODELO ========== #
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ========== ENTRENAMIENTO ========== #
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# ========== EVALUACIÓN ========== #
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n🎯 Accuracy en test: {test_accuracy:.4f}")

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

print("\n📊 Reporte de clasificación:\n")
print(classification_report(test_generator.classes, y_pred, target_names=test_generator.class_indices.keys()))

# ========== VISUALIZACIÓN ========== #
def plot_metrics(history):
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión (accuracy)')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.grid(True)
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida (loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

plot_metrics(history)
