#  Clasificación de Alzheimer usando MobileNetV2 + Fine-Tuning

Este proyecto implementa un modelo de deep learning basado en **MobileNetV2**, una arquitectura del estado del arte, para clasificar imágenes médicas en cuatro categorías relacionadas con el Alzheimer:  
- **MildDemented**  
- **ModerateDemented**  
- **NonDemented**  
- **VeryMildDemented**

---

##  Modelo del Estado del Arte

El modelo se basa en el artículo:

> **Howard et al., 2018**  
> _"MobileNetV2: Inverted Residuals and Linear Bottlenecks"_  
>  [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)

MobileNetV2 ha sido validado en tareas de clasificación de imágenes con recursos computacionales limitados y se ha utilizado con éxito en diagnósticos médicos por imagen, incluyendo enfermedades neurológicas.

---

## Implementación

### Frameworks usados
- `TensorFlow` y `Keras` para la implementación del modelo
- `Matplotlib` y `Seaborn` para la visualización de resultados
- `Scikit-learn` para métricas de evaluación

### Técnicas aplicadas
- **Transfer Learning**: usando pesos preentrenados de ImageNet
- **Fine-Tuning**: se desbloquearon las últimas 30 capas del modelo base
- **Data Augmentation**: se aplicaron rotaciones, zoom y flips horizontales para mejorar la generalización
- **Dropout Regularization**: para evitar overfitting
- **Early Stopping y Model Checkpointing**: para guardar el mejor modelo

---

##  Evaluación del Modelo

Se utilizaron las siguientes métricas, recomendadas en la literatura para problemas de clasificación médica:

- **Accuracy** (Precisión global)
- **Precision**
- **Recall** (Sensibilidad)
- **F1-score**
- **Matriz de Confusión**

> Basado en prácticas recomendadas en:  
> *“Deep Learning for Alzheimer’s Disease Classification”*  
> [PMCID: PMC7875387](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7875387/)

---

## Resultados Obtenidos

### Métricas globales del conjunto de prueba:

```
Accuracy:     0.61
Precision:    0.64
Recall:       0.61
F1-score:     0.56
```

### Matriz de Confusión:

La mayor precisión se obtuvo en la clase **NonDemented**, con mejoras notables en **VeryMildDemented**. Las clases menos representadas (**ModerateDemented**) aún presentan dificultades debido al desbalance de datos.

---

## Curvas de entrenamiento

- Las curvas muestran una tendencia **estable y creciente**, sin señales evidentes de **overfitting**.
- El modelo **generaliza razonablemente bien**.

---

## Próximos pasos

- Balancear las clases mediante `class_weights` o `SMOTE`.
- Probar otras arquitecturas del estado del arte como EfficientNet.
- Aplicar interpretabilidad con Grad-CAM para visualizar las regiones de atención del modelo.
- Hacer validación cruzada para resultados más robustos.

---

## Conclusión

El modelo MobileNetV2 adaptado con fine-tuning y técnicas de regularización logra un rendimiento aceptable en la clasificación del Alzheimer. Aunque existe margen de mejora en clases menos representadas, el enfoque basado en el estado del arte demuestra ser eficaz y escalable.
