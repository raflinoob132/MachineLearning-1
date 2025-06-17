# Pipeline Machine Learning untuk Klasifikasi Gambar Ikan
# Tugas: Prediksi Jenis Ikan menggunakan CNN

# =====================================
# 1. IMPORT LIBRARIES DAN SETUP
# =====================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# TensorFlow dan Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
# Metrics dan Evaluasi
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools

# Mount Google Drive (jalankan dan ikuti instruksi untuk authorization)
#from google.colab import drive
#drive.mount('/content/drive')

# Pastikan Anda telah mengunduh dataset dari Kaggle secara manual atau menggunakan Kaggle API.
# Contoh perintah terminal untuk mengunduh dataset:
# !kaggle datasets download -d markdaniellampa/fish-dataset
# Kemudian ekstrak file zip ke direktori yang diinginkan.

# Misalkan dataset sudah diekstrak ke folder 'fish-dataset' di direktori kerja Anda
DATASET_PATH = 'archive\FishImgDataset'  # Ganti dengan path sesuai lokasi ekstraksi dataset Anda

print("Path to dataset files:", DATASET_PATH)
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
VAL_PATH = os.path.join(DATASET_PATH, 'val')
TEST_PATH = os.path.join(DATASET_PATH, 'test')

# Parameter model
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

print("Setup selesai!")


# =====================================
# 3. EKSPLORASI DAN PEMBERSIHAN DATA
# =====================================

def explore_dataset(data_path):
    """Fungsi untuk eksplorasi dataset"""
    if not os.path.exists(data_path):
        print(f"Path {data_path} tidak ditemukan!")
        return None

    classes = os.listdir(data_path)
    print(f"Jumlah kelas: {len(classes)}")
    print(f"Nama kelas: {classes}")

    class_counts = {}
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts[class_name] = count

    return class_counts

def create_data_summary():
    """Membuat ringkasan data untuk analisis"""
    print("=== EKSPLORASI DATASET ===")

    train_data = explore_dataset(TRAIN_PATH)
    val_data = explore_dataset(VAL_PATH)
    test_data = explore_dataset(TEST_PATH)

    if train_data:
        # Buat DataFrame untuk visualisasi
        df_train = pd.DataFrame(list(train_data.items()), columns=['Species', 'Count'])
        df_train['Dataset'] = 'Train'

        if val_data:
            df_val = pd.DataFrame(list(val_data.items()), columns=['Species', 'Count'])
            df_val['Dataset'] = 'Validation'

        if test_data:
            df_test = pd.DataFrame(list(test_data.items()), columns=['Species', 'Count'])
            df_test['Dataset'] = 'Test'

        # Gabungkan data
        if val_data and test_data:
            df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)
        else:
            df_combined = df_train

        # Visualisasi distribusi data
        plt.figure(figsize=(15, 6))
        sns.barplot(data=df_combined, x='Species', y='Count', hue='Dataset')
        plt.title('Distribusi Data per Spesies Ikan')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return df_combined

    return None

# Jalankan eksplorasi data
data_summary = create_data_summary()


# =====================================
# 4. PREPROCESSING DAN AUGMENTASI DATA
# =====================================

def create_data_generators():
    """Membuat data generators dengan augmentasi"""

    # Data Augmentation untuk training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1
    )

    # Hanya rescaling untuk validation dan test
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Generator untuk training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    # Generator untuk validation data
    validation_generator = val_test_datagen.flow_from_directory(
        VAL_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    # Generator untuk test data
    test_generator = val_test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_generator, validation_generator, test_generator

# Buat data generators
train_gen, val_gen, test_gen = create_data_generators()

# Info tentang kelas
class_names = list(train_gen.class_indices.keys())
num_classes = len(class_names)
print(f"Jumlah kelas: {num_classes}")
print(f"Nama kelas: {class_names}")


# =====================================
# 5. FEATURE ENGINEERING & LABEL ENCODING
# =====================================

def visualize_sample_images(generator, class_names):
    """Visualisasi sample gambar dari setiap kelas"""
    plt.figure(figsize=(15, 10))

    # Ambil satu batch data
    batch_images, batch_labels = next(generator)

    for i in range(min(16, len(batch_images))):
        plt.subplot(4, 4, i + 1)
        plt.imshow(batch_images[i])

        # Dapatkan label kelas
        class_idx = np.argmax(batch_labels[i])
        plt.title(f'Kelas: {class_names[class_idx]}')
        plt.axis('off')

    plt.suptitle('Sample Gambar dari Dataset')
    plt.tight_layout()
    plt.show()

# Visualisasi sample gambar
visualize_sample_images(train_gen, class_names)

# Reset generator setelah visualisasi
train_gen.reset()

# =====================================
# 6. MODEL BUILDING - CNN ARCHITECTURE
# =====================================

def create_cnn_model(input_shape, num_classes):
    """Model CNN custom"""
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def create_transfer_learning_model(base_model_name, input_shape, num_classes):
    """Model dengan Transfer Learning"""

    # Pilih base model
    if base_model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze base model layers
    base_model.trainable = False

    # Tambahkan classifier layers
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Buat berbagai model
input_shape = (224, 224, 3)

print("Membuat model...")
cnn_model = create_cnn_model(input_shape, num_classes)
vgg_model = create_transfer_learning_model('vgg16', input_shape, num_classes)
resnet_model = create_transfer_learning_model('resnet50', input_shape, num_classes)

# Compile models
models = {
    'CNN_Custom': cnn_model,
    'VGG16_Transfer': vgg_model,
    # 'ResNet50_Transfer': resnet_model
}

for name, model in models.items():
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Model {name} berhasil di-compile")


def train_model(model, model_name, train_gen, val_gen):
    checkpoint_path = f'{model_name}_checkpoint.weights.h5'
    log_path = f"{model_name}_training_log.csv"

    # Cek apakah training sudah selesai
    if os.path.exists(log_path):
        import pandas as pd
        log = pd.read_csv(log_path)
        if len(log) >= EPOCHS:
            print("Training sudah selesai, skip training.")
            # Buat objek mirip history
            class DummyHistory:
                def __init__(self, log):
                    self.history = log.to_dict(orient='list')
            return DummyHistory(log)
        # Coba load checkpoint jika log belum penuh
        if os.path.exists(checkpoint_path):
            print("Checkpoint found, loading weights...")
            model.load_weights(checkpoint_path)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
        ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=False),
        CSVLogger(f'{model_name}_training_log.csv', append=True)
    ]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights

    )

    return history


# Training semua model
histories = {}
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"TRAINING MODEL: {name}")
    print(f"{'='*50}")

    history = train_model(model, name, train_gen, val_gen)
    histories[name] = history


# =====================================
# 8. EVALUASI MODEL DAN METRICS
# =====================================

def plot_training_history(histories):
    """Plot training history untuk semua model"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History Comparison', fontsize=16)

    # Accuracy
    axes[0, 0].set_title('Training Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[1, 0].set_title('Training Loss')
    axes[1, 1].set_title('Validation Loss')

    for name, history in histories.items():
        epochs = range(1, len(history.history['accuracy']) + 1)

        axes[0, 0].plot(epochs, history.history['accuracy'], label=name)
        axes[0, 1].plot(epochs, history.history['val_accuracy'], label=name)
        axes[1, 0].plot(epochs, history.history['loss'], label=name)
        axes[1, 1].plot(epochs, history.history['val_loss'], label=name)

    for ax in axes.flat:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_generator, class_names):
    """Evaluasi model dengan berbagai metrics"""

    # Prediksi
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # True labels
    true_classes = test_generator.classes

    # Metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    f1 = f1_score(true_classes, predicted_classes, average='weighted')

    # Classification report
    report = classification_report(true_classes, predicted_classes,
                                 target_names=class_names, output_dict=True)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'classification_report': report
    }

def plot_confusion_matrix(true_classes, predicted_classes, class_names, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def calculate_auc_roc(true_classes, predictions, class_names):
    """Hitung AUC-ROC untuk multi-class"""
    # Convert to binary format
    lb = LabelBinarizer()
    true_binary = lb.fit_transform(true_classes)

    # Jika hanya 2 kelas, reshape
    if len(class_names) == 2:
        true_binary = np.hstack((1 - true_binary, true_binary))

    # Hitung AUC untuk setiap kelas
    auc_scores = {}

    plt.figure(figsize=(12, 8))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(true_binary[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = roc_auc

        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Hitung macro average
    macro_auc = np.mean(list(auc_scores.values()))

    return auc_scores, macro_auc

# Plot training history
plot_training_history(histories)

# Evaluasi semua model
print("\n" + "="*60)
print("EVALUASI MODEL")
print("="*60)

evaluation_results = {}

for name, model in models.items():
    print(f"\nEvaluasi Model: {name}")
    print("-" * 40)

    # Evaluasi
    results = evaluate_model(model, test_gen, class_names)
    evaluation_results[name] = results

    # Print metrics
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")

    # Confusion Matrix
    plot_confusion_matrix(results['true_classes'], results['predicted_classes'],
                         class_names, name)

    # AUC-ROC
    auc_scores, macro_auc = calculate_auc_roc(results['true_classes'],
                                             results['predictions'], class_names)
    print(f"Macro AUC-ROC: {macro_auc:.4f}")

    evaluation_results[name]['auc_roc'] = macro_auc

# =====================================
# 9. PERBANDINGAN HASIL DAN ANALISIS
# =====================================

def create_comparison_table(evaluation_results):
    """Membuat tabel perbandingan semua model"""

    comparison_data = []
    for name, results in evaluation_results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'AUC-ROC': results['auc_roc']
        })

    df_comparison = pd.DataFrame(comparison_data)

    # Sort berdasarkan F1-Score
    df_comparison = df_comparison.sort_values('F1-Score', ascending=False)

    return df_comparison

def plot_metrics_comparison(df_comparison):
    """Plot perbandingan metrics"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.barplot(data=df_comparison, x='Model', y=metric, ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].tick_params(axis='x', rotation=45)

        # Tambahkan nilai di atas bar
        for j, v in enumerate(df_comparison[metric]):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # Hapus subplot kosong
    axes[5].remove()

    plt.tight_layout()
    plt.show()

# Buat perbandingan
comparison_table = create_comparison_table(evaluation_results)
print("\nTABEL PERBANDINGAN MODEL:")
print("="*50)
print(comparison_table.round(4))

# Plot perbandingan
plot_metrics_comparison(comparison_table)

# =====================================
# 10. ANALISIS DAN REKOMENDASI
# =====================================

def analyze_results(comparison_table, evaluation_results):
    """Analisis hasil dan memberikan rekomendasi"""

    best_model = comparison_table.iloc[0]['Model']
    best_results = evaluation_results[best_model]

    print(f"\n{'='*60}")
    print("ANALISIS HASIL DAN REKOMENDASI")
    print(f"{'='*60}")

    print(f"\n🏆 MODEL TERBAIK: {best_model}")
    print(f"   - Accuracy: {best_results['accuracy']:.4f}")
    print(f"   - F1-Score: {best_results['f1_score']:.4f}")
    print(f"   - AUC-ROC: {best_results['auc_roc']:.4f}")

    print(f"\n📊 PENJELASAN METRICS:")
    print(f"   • ACCURACY: Mengukur proporsi prediksi yang benar dari total prediksi")
    print(f"   • PRECISION: Mengukur ketepatan prediksi positif")
    print(f"   • RECALL: Mengukur kemampuan model mendeteksi kelas positif")
    print(f"   • F1-SCORE: Harmonic mean dari precision dan recall")
    print(f"   • AUC-ROC: Mengukur kemampuan model membedakan antar kelas")

    print(f"\n🎯 METRIC TERBAIK UNTUK KLASIFIKASI IKAN:")
    print(f"   F1-Score adalah metric terbaik karena:")
    print(f"   1. Menyeimbangkan precision dan recall")
    print(f"   2. Cocok untuk multi-class classification")
    print(f"   3. Robust terhadap class imbalance")
    print(f"   4. Memberikan gambilan performa keseluruhan yang baik")

    print(f"\n💡 REKOMENDASI:")
    print(f"   1. Gunakan model {best_model} untuk deployment")
    print(f"   2. Monitor performa secara berkala")
    print(f"   3. Kumpulkan lebih banyak data untuk improvement")
    print(f"   4. Lakukan fine-tuning jika diperlukan")

# Jalankan analisis
analyze_results(comparison_table, evaluation_results)

print(f"\n{'='*60}")
print("PIPELINE SELESAI! 🎉")
print(f"{'='*60}")
print("Semua model telah dilatih dan dievaluasi.")
print("Silakan gunakan model terbaik untuk prediksi ikan Anda!")
