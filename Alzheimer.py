import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm

# CONFIGURACIÓN
IMG_SIZE = (96, 96)
DATASET_DIR = r"C:\Users\dafne\OneDrive\Documentos\Escritorio\Alzheimer\Alzheimer_s Dataset\test"
OUTPUT_DIR = "output_errores"
MODEL_PATH = "modelo_alzheimer.keras"
LAST_CONV_LAYER = "Conv_1"

# Crear carpeta si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = load_model(MODEL_PATH)
class_names = sorted(os.listdir(DATASET_DIR))

def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array /= 255.
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), predictions.numpy()

def save_gradcam(img_path, heatmap, prediction, true_class):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    base_name = os.path.basename(img_path).replace(" ", "_")
    out_name = f"{true_class}_pred-{predicted_class}_{confidence:.2f}_{base_name}"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, superimposed_img)

# PROCESAMIENTO
print("🔍 Filtrando solo errores de predicción...\n")

for class_dir in class_names:
    class_path = os.path.join(DATASET_DIR, class_dir)
    image_files = os.listdir(class_path)  # Puedes limitar a [:6] si es solo demo

    for img_file in tqdm(image_files, desc=f"{class_dir}", unit="img"):
        img_path = os.path.join(class_path, img_file)
        try:
            img_array = get_img_array(img_path, IMG_SIZE)
            heatmap, prediction = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
            predicted_class = class_names[np.argmax(prediction)]

            if predicted_class != class_dir:  # SOLO SI ES ERROR
                save_gradcam(img_path, heatmap, prediction, class_dir)

        except Exception as e:
            print(f"⚠️ Error en {img_path}: {e}")

# MOSTRAR EN GALERÍA
print("\n📸 Mostrando SOLO errores en galería por páginas...\n")

output_images = sorted(os.listdir(OUTPUT_DIR))
images_per_page = 16
cols = 4

for start in range(0, len(output_images), images_per_page):
    end = min(start + images_per_page, len(output_images))
    page_images = output_images[start:end]
    rows = (len(page_images) // cols) + int(len(page_images) % cols != 0)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    for i, img_file in enumerate(page_images):
        path = os.path.join(OUTPUT_DIR, img_file)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = axs[i // cols, i % cols] if rows > 1 else axs[i]
        ax.imshow(img_rgb)
        ax.set_title(img_file[:60], fontsize=8)
        ax.axis('off')

    for j in range(len(page_images), rows * cols):
        ax = axs[j // cols, j % cols] if rows > 1 else axs[j]
        ax.axis('off')

    plt.tight_layout()
    plt.show()
