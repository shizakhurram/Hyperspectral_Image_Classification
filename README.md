## 🌾 Hyperspectral Image Classification
A comprehensive implementation of Convolutional Neural Networks (CNNs) for classifying hyperspectral images (HSI), focusing on the **Indian Pines** and **Pavia University** datasets. The pipeline includes preprocessing, class balancing, augmentation, model training, and rich visualizations.

---

### 🔧 Features

- **Dataset Support:**  
  - Indian Pines (`freeman2147/1992-indian-pines`)  
  - Pavia University (`mlxlx0000/paviadata`)

- **Preprocessing:**  
  - Patch creation via sliding window (default: 5×5)  
  - Zero padding for borders  
  - Option to discard zero-label patches

- **Class Balancing:**  
  - Oversampling of minority classes  
  - Random permutation for robust training

- **Data Augmentation:**  
  - Horizontal & vertical flipping  
  - Random rotations (±180° in 30° steps)

---

### 🧠 Model Architecture

A lightweight CNN model:

```python
model = Sequential([
    Conv2D(C1, (3, 3), activation='relu'),
    Conv2D(3*C1, (3, 3), activation='relu'),
    Dropout(0.25),
    Flatten(),
    Dense(6*numComponents, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='softmax')
])
```

- Optimizer: `Adam`  
- Loss: `Categorical Crossentropy`  
- Batch Size: `32`  
- Customizable epochs

---

### 📊 Evaluation

- Accuracy, Loss, Confusion Matrix  
- Class-wise Precision, Recall, F1-Score  
- ROC & Precision-Recall Curves  
- t-SNE Embedding Visuals

---

### 🖼️ Visualization Tools

- **Data-Level:**  
  - False color composites  
  - Ground truth maps  
  - Spectral signature plots

- **Training-Level:**  
  - Accuracy/loss curves  
  - Class-wise performance plots  
  - Batch visualizations

- **Results-Level:**  
  - Classification maps  
  - Confusion heatmaps  
  - CAMs and feature maps

---

### ✅ Classification Targets

Supports 16 land cover types including:

- Alfalfa, Corn (various), Soybeans (various), Grass types, Hay, Oats, Wheat, Woods, and Man-made surfaces (Buildings, Steel Towers)

---

### 🚀 Usage Overview

```python
# Patch creation
XPatches, yPatches = createPatches(indian_pines_pca, indian_pines_gt, windowSize=5)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(XPatches, yPatches, test_size=0.2)

# Balance & augment
X_train, y_train = oversampleWeakClasses(X_train, y_train)
X_train = AugmentData(X_train)

# Model training
model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS)

# Evaluation
classification, confusion, test_Loss, test_accuracy = reports(X_test, y_test)

# Visualization
plot_confusion_matrix(confusion)
plot_classification_map(prediction_map, ground_truth)
plot_feature_maps(model, X_test[0])
```

---

### 🎨 Visualization Best Practices

- **Colors:** Use colorblind-friendly, consistent palettes  
- **Layout:** Label axes, use colorbars, proper sizing  
- **Interactivity:** Zoom, tooltips, toggle views  
- **Export:** PNG, PDF, SVG with high resolution & scale bars
