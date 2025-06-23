````markdown
# 🐶 Dog Breed Classifier with EfficientNetB0

A deep learning model for classifying 120 dog breeds using EfficientNetB0 and transfer learning. Built with TensorFlow and Keras.

---

## 📁 Folder Structure

```plaintext
dog-breed-classifier/
├── data/
│   ├── labels.csv              # Contains image filenames and dog breed labels
│   └── train/                  # Directory with all training images (e.g., .jpg files)
│       ├── 001d0874beef0a3c.jpg
│       ├── 00a338a92e4e7bf5.jpg
│       └── ...
├── dog_classifier.py           # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
````

---

## 📊 Dataset

* **Source**: [Kaggle Dog Breed Identification Dataset](https://www.kaggle.com/c/dog-breed-identification/data)
* `labels.csv` should contain:

```csv
id,class
001d0874beef0a3c,affenpinscher
00a338a92e4e7bf5,afghan_hound
...
```

* The script automatically appends `.jpg` to the filenames.

---

## 🧠 Model Architecture

| Component   | Details                                  |
| ----------- | ---------------------------------------- |
| Base Model  | EfficientNetB0 (pretrained on ImageNet)  |
| Top Layers  | GlobalAvgPool → Dense(128) → Dropout     |
| Final Layer | Dense(num\_classes) with softmax         |
| Optimizer   | Adam                                     |
| Loss        | Categorical Crossentropy                 |
| Strategy    | Freeze base → train top → fine-tune base |

---

## 🔄 Data Augmentation (Training Only)

* Horizontal & vertical flips
* Zoom (30%)
* Rotation (±30°)
* Brightness (±30%)
* Shear, width & height shift

Used via `ImageDataGenerator` to improve generalization.

---

## ⚖️ Class Weight Handling

Class imbalance is addressed using `compute_class_weight()` from `sklearn`:

* Automatically balances dog breed distribution
* Correctly maps class **names to numeric indices**

✅ No manual editing needed.

---

## ✅ Callbacks

* `EarlyStopping`: Stops when validation loss stops improving
* `ReduceLROnPlateau`: Reduces learning rate if model plateaus

---

## 🧪 Optional Test Set

You can split off 10% of the data for final unbiased testing:

```python
train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df['class'])
```

Then evaluate after training:

```python
model.evaluate(test_generator)
```

---

## 🚀 How to Use

1. **Clone the repo**

```bash
git clone https://github.com/your-username/dog-breed-classifier.git
cd dog-breed-classifier
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**

* Download from Kaggle and place `labels.csv` and `train/` inside the `data/` folder.

4. **Run the training**

```bash
python dog_classifier.py
```

5. **Trained model will be saved as:**

```bash
fine.h5
```

---

## 📦 Requirements

* TensorFlow ≥ 2.9
* pandas
* numpy
* scikit-learn
* matplotlib (optional)

Install with:

```bash
pip install -r requirements.txt
```

---

## 👤 Author

Made with 💙 by **Aryan Gupta**
📧 Feel free to open issues, fork, or suggest improvements.

---

## 📄 License

MIT License – free to use and modify.

````

---

### ✅ Bonus Files You Should Add

1. **`requirements.txt`**  
```txt
tensorflow
pandas
numpy
scikit-learn
matplotlib
````

2. **`.gitignore`**

```txt
*.h5
__pycache__/
.ipynb_checkpoints/
.env
.DS_Store
```

