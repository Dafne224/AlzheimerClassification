# Clasificación de Alzheimer con MobileNetV2

Este proyecto busca construir un modelo de deep learning capaz de clasificar imágenes de resonancias magnéticas cerebrales para detectar distintos estados del Alzheimer. Está orientado a apoyar el diagnóstico médico automatizado, facilitando una detección temprana y precisa de esta enfermedad neurodegenerativa.

---

## Objetivos del Proyecto

- Clasificar imágenes de resonancia magnética cerebral en cuatro categorías clínicas:  
  - **NonDemented**  
  - **VeryMildDemented**  
  - **MildDemented**  
  - **ModerateDemented**

- Probar modelos del estado del arte aplicados al dominio biomédico.
- Mejorar el rendimiento a través de **fine-tuning**, aumento de datos y ajuste de hiperparámetros.

---

## Modelo Usado: MobileNetV2

El modelo central es **MobileNetV2**, una red convolucional profunda eficiente y ligera, diseñada para ejecutarse en dispositivos con recursos limitados. Está basada en:

> **MobileNetV2: Inverted Residuals and Linear Bottlenecks**  
> *Mark Sandler, Andrew Howard, et al. — Google AI (2018)*  
> [🔗 Paper en arXiv](https://arxiv.org/abs/1801.04381)

MobileNetV2 ofrece un buen equilibrio entre precisión y eficiencia computacional, y ha sido usado exitosamente en tareas de clasificación médica.

---

## Otros artículos relacionados

- **"Alzheimer Disease Detection Using Transfer Learning with Convolutional Neural Networks"**  
  [DOI: 10.1109/ACCESS.2020.3012332](https://ieeexplore.ieee.org/document/9144412)
- **"Deep Learning for Alzheimer’s Disease Classification: A Comparative Analysis"**  
  [DOI: 10.1016/j.neunet.2019.12.001](https://www.sciencedirect.com/science/article/pii/S0893608019302506)

---

## ⚙️ Implementación

El proyecto fue desarrollado con **TensorFlow + Keras**. Se utilizaron técnicas de:

- Aumento de datos (data augmentation)
- Transfer learning (MobileNetV2 preentrenada en ImageNet)
- Fine-tuning (ajuste fino de las últimas 30 capas)
- Dropout para prevenir overfitting

### Arquitectura Personalizada

- `GlobalAveragePooling2D`
- `Dropout(0.3)`
- `Dense(512, ReLU)`
- `Dropout(0.3)`
- `Dense(4, Softmax)`

---

## Dataset

Conjunto de datos: **Alzheimer's MRI Dataset**  
Dividido en:

- `train/` → 4098 imágenes  
- `val/` → 1023 imágenes  
- `test/` → 1279 imágenes  

---

## Resultados de la Primera Iteración

La primera versión del modelo usó solo la base congelada de MobileNetV2 sin fine-tuning.

| Métrica     | Valor     |
|-------------|-----------|
| Accuracy    | 54.34%    |
| Precision   | 0.51      |
| Recall      | 0.58      |
| F1-score    | 0.51      |

 Se observó un sesgo hacia la clase **NonDemented**, con bajo rendimiento en clases minoritarias.

---

## Resultados de la Iteración Mejorada (Fine-tuning y más capas)

Se aplicó fine-tuning (desbloqueo de últimas 30 capas), más complejidad en las capas densas, y mayor número de épocas.

### Reporte de Clasificación

| Clase              | Precision | Recall | F1-score | Soporte |
|--------------------|-----------|--------|----------|---------|
| MildDemented       | 1.00      | 0.05   | 0.10     | 179     |
| ModerateDemented   | 0.00      | 0.00   | 0.00     | 12      |
| NonDemented        | 0.57      | 0.96   | 0.71     | 640     |
| VeryMildDemented   | 0.64      | 0.27   | 0.38     | 448     |

| Promedio           | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Macro average      | 0.58     | 0.55      | 0.32   | 0.30     |
| Weighted average   | 0.58     | 0.65      | 0.58   | 0.50     |

**Precisión en el set de prueba: 57.94%**

 Mejora notable en las métricas globales, aunque aún se identifican desafíos en las clases menos representadas.

---

## Visualizaciones

### 🔷 Matriz de Confusión

![Matriz de Confusión](ruta/a/tu/imagen_confusion.png)

### Curvas de Precisión y Pérdida

![Gráficas de Entrenamiento](ruta/a/tu/imagen_curvas.png)

---

## Interpretación de Resultados

- El modelo mejoró la precisión general con fine-tuning y capas densas personalizadas.
- Se sigue observando un **desbalance** en las clases, especialmente **ModerateDemented**, lo que impacta las métricas macro.
- El uso de MobileNetV2 permitió mantener un entrenamiento eficiente y tiempos razonables.

---

## Referencias

1. Sandler, M., et al. *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"*. arXiv, 2018.  
2. Jain, R., et al. *"Alzheimer Disease Detection Using Transfer Learning"*. IEEE Access, 2020.  
3. Basaia, S., et al. *"Automated Classification of Alzheimer’s Disease and Mild Cognitive Impairment Using Deep Learning"*. Neurolmage: Clinical, 2019.

---

## 🚀 Cómo ejecutar

```bash
python Modelo_Alzheimer.py
