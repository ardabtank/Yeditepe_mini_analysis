'''
Importing necessary libraries and modules for data processing,
model creation, and evaluation which are important for any machine learning project.
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras import regularizers # type: ignore
import pickle
import matplotlib.pyplot as plt
from tabulate import tabulate
from IPython.display import display, HTML
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, matthews_corrcoef, log_loss, roc_curve
import os
import plotly.graph_objects as go #type: ignore
from tensorflow.keras.layers import LeakyReLU # type: ignore
import json
import tensorflow_datasets as tfds
import tempfile
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import RobustScaler
import shap


'''
Class ColorPrinter provides methods to
print colored text in HTML format.
'''
class ColorPrinter:
    @staticmethod
    def print_red(text):
        return f"<span style='color:red; font-size: 16px;'>{text}</span>"

    @staticmethod
    def print_orange(text):
        return f"<span style='color:orange; font-size: 16px;'>{text}</span>"

    @staticmethod
    def print_green(text):
        return f"<span style='color:green; font-size: 16px;'>{text}</span>"

    @staticmethod
    def print_yellow(text):
        return f"<span style='color:yellow; font-size: 16px;'>{text}</span>"

    @staticmethod
    def print_blue(text):
        return f"<span style='color:blue; font-size: 16px;'>{text}</span>"

    @staticmethod
    def print_magenta(text):
        return f"<span style='color:magenta; font-size: 16px;'>{text}</span>"

    @staticmethod
    def print_cyan(text):
        return f"<span style='color:cyan; font-size: 16px;'>{text}</span>"


'''
Class DataProcessor includes the processes of loading, balancing
and preprocessing the loaded dataset. The datasets we'll use here are
analyzed and collected from ATLAS 13 TeV Open Data.
'''

class DataProcessor:
    def __init__(self, file_paths, cache_file='cached_data.pkl'):
        self.file_paths = file_paths
        self.df = None
        self.scaler = StandardScaler()
        self.cache_file = cache_file

    def load_data(self):
        if os.path.exists(self.cache_file):
            print(f"{self.cache_file} is found, loading from cache...")
            self.df = pd.read_pickle(self.cache_file)
        else:
            print("Cache couldn't be found, loading from CSV file...")
            dataframes = [pd.read_csv(file_path) for file_path in self.file_paths]
            self.df = pd.concat(dataframes, ignore_index=True)
            self.df.to_pickle(self.cache_file)
            print(f"Veri {self.cache_file} dosyasına cachelendi.")

        X = self.df.drop(['DATA_MC_CHECK', 'Xsection'], axis=1)
        y = self.df['DATA_MC_CHECK']

        feature_names=X.columns
        print("Feature names:", feature_names.to_list)

        if os.path.exists('processed_data_with_weights.pkl'):
            print("Weights are loading from the pickle file.")
            self.df = pd.read_pickle('processed_data_with_weights.pkl')
            weights = self.df['weights']
        else:
            print("Weights calculating...")
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
            class_weight_dict = dict(enumerate(class_weights))
            print("Class Weights:", class_weight_dict)

            #Multiplying both Xsection weights and class weights and getting a final weight
            self.df['weights'] = self.df.apply(lambda row: row['Xsection'] * class_weight_dict[row['DATA_MC_CHECK']], axis=1)
            self.df.to_pickle('processed_data_with_weights.pkl')
            print("Weights have calculated and saved in pickle.")
            weights = self.df['weights']

        X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
            X, y, weights, test_size=0.2, stratify=y, random_state=42)

        X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
            X_temp, y_temp, w_temp, test_size=0.005, stratify=y_temp, random_state=42)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # Normalize işlemi için scaler'ı kaydet
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        return (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test)

    def preprocess_data(self, X, y, w):
        dataset = tf.data.Dataset.from_tensor_slices((X, y, w))

        BUFFER_SIZE = 10000
        BATCH_SIZE = 16384

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        # Cache işlemi
        dataset = dataset.cache()

        # Prefetching
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

class ModelBuilder:
    def __init__(self):
        self.model = None

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(12, kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(12,)),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])


        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')],
            weighted_metrics=[
                tf.keras.metrics.BinaryAccuracy(name='weighted_accuracy'),
                tf.keras.metrics.AUC(name='weighted_auc'),
                tf.keras.metrics.Precision(name='weighted_precision'),
                tf.keras.metrics.Recall(name='weighted_recall')]
)
        return self.model


if __name__ == "__main__":
    file_paths = ['/content/drive/MyDrive/Zanaliz/8 agustus 2024/ZPrime.csv', '/content/drive/MyDrive/Zanaliz/8 agustus 2024/background.csv']  # Dosya yolunu buraya ekle
    data_processor = DataProcessor(file_paths)

    (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test) = data_processor.load_data()

    # Preprocessing of datasets
    train_dataset = data_processor.preprocess_data(X_train, y_train, w_train)
    validation_dataset = data_processor.preprocess_data(X_val, y_val, w_val)

    #Preprocess and saving the unseen test split as pickle.
    test_dataset = data_processor.preprocess_data(X_test, y_test, w_test)
    with open('test_dataset.pkl', 'wb') as f:
      pickle.dump((X_test, y_test, w_test), f)

    model_builder = ModelBuilder()
    model = model_builder.build_model()

    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        epochs=100,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    # Modeli kaydetme
    model.save('ZPrime_prediction_model.keras')

    model.summary()

    model_config = model.get_config()

    json_path = "model_config.json"
    with open(json_path, 'w') as json_file:
        json.dump(model_config, json_file)

'''
Evaluation of the model with validation set is below.

Some performance metrics are shown with explanations and graphs are plotted.
Then the graphs are saved as png file into a folder named Graphs.
'''

display(HTML(ColorPrinter.print_yellow("<br><br>EVALUATION OF THE MODEL:<br><br>")))

def create_shap_summary_plot(model, X_train, X_val):
    try:
        import shap

        # Prepare background dataset
        background_indices = np.random.choice(X_train.shape[0], size=100, replace=False)
        background = X_train[background_indices]

        # Create SHAP explainer
        explainer = shap.GradientExplainer(model, background)

        # Compute SHAP values
        sample_to_explain = X_val[:100]
        shap_values = explainer.shap_values(sample_to_explain)

        # Feature names
        feature_names = ['el0_E', 'el0_phi', 'el0_eta', 'pt_el0', 'el1_E', 'el1_phi', 'el1_eta',
                         'pt_el1', 'delta_R', 'jet_n', 'met_et', 'HT']

        # Plot summary
        shap.summary_plot(shap_values[0], sample_to_explain, feature_names=feature_names, show=False)
        plt.savefig('shap_summary_plot.png', bbox_inches='tight')
        plt.close()

        print("SHAP summary plot successfully created and saved.")
    except Exception as e:
        print("Failed to create SHAP summary plot. An error occurred.")
        print(str(e))

create_shap_summary_plot(model, X_train, X_val)


# DEFINING THE LOSS OF TRAIN AND TEST
loss = history.history['loss']
val_loss = history.history['val_loss']

# DEFINING THE ACCURACY OF TRAIN AND TEST
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)

# THE GRAPH OF TRAINING AND VALIDATION LOSS
display(HTML(ColorPrinter.print_orange("<br><br>TRAINING AND VALIDATION LOSS:<br><br>")))
plt.figure(figsize=(12, 5))
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.legend()
plt.title('Training and Validation Loss')
# Save the figure with higher resolution
plt.savefig('training_and_validation_loss.png', dpi=300)
plt.show()

# THE GRAPH OF TRAINING AND VALIDATION ACCURACY
display(HTML(ColorPrinter.print_orange("<br><br>TRAINING AND VALIDATION ACCURACY:<br><br>")))
plt.figure(figsize=(12, 5))
plt.plot(epochs, train_accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_and_validation_accuracy.png', dpi=300)
plt.show()

#LOSS AND ACCURACY ON UNSEEN TEST DATA:

validation_dataset_X = tf.concat([x for x, _, __ in validation_dataset], axis=0)
validation_dataset_y = tf.concat([y for _, y, __ in validation_dataset], axis=0)
validation_dataset_w = tf.concat([w for _, __, w in validation_dataset], axis=0)

y_pred = model.predict(validation_dataset_X, batch_size=512)

y_pred_classes = (y_pred > 0.5).astype(int)

if not isinstance(y_pred, tf.Tensor):
    y_pred = tf.convert_to_tensor(y_pred)

# Neural Network output
# Ensure consistent data types
validation_dataset_y = tf.cast(validation_dataset_y, tf.float32)
y_pred = tf.cast(y_pred, tf.float32)

# Define the threshold as a scalar
threshold = 0.5

# Create boolean masks
signal_mask = tf.greater(validation_dataset_y, threshold)
background_mask = tf.less_equal(validation_dataset_y, threshold)

# Apply masks to predictions
signal_decisions = tf.boolean_mask(y_pred, signal_mask)
background_decisions = tf.boolean_mask(y_pred, background_mask)

# Convert tensors to NumPy arrays for plotting
signal_decisions_np = signal_decisions.numpy()
background_decisions_np = background_decisions.numpy()

# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(background_decisions_np, bins=50, color='red', label='Background',
         histtype='step', density=True, linewidth=1.5)
plt.hist(signal_decisions_np, bins=50, color='blue', label='Signal',
         histtype='step', density=True, linestyle='--', linewidth=1.5)

plt.xlabel('Neural Network Output')
plt.ylabel('Density')
plt.title('Neural Network Output Distribution for Signal and Background')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model on the validation dataset
eval_results = model.evaluate(validation_dataset)
print(f"Validation Metrics: {eval_results}")


# Generate classification report
report_dict = classification_report(validation_dataset_y, y_pred_classes, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
# Display as a table
display(report_df.style.background_gradient(cmap='viridis'))



# THE PROCESS OF PERFORMANCE METRICS
display(HTML(ColorPrinter.print_yellow("<br><br>PERFORMANCE METRICS:<br><br>")))

# ROC Curve
def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)  # scikit-learn'ün roc_curve fonksiyonu
    plt.plot(fpr, tpr, linestyle='--', label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)  # scikit-learn'ün precision_recall_curve fonksiyonu
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.show()

y_true_np = validation_dataset_y.numpy()
y_pred_np = y_pred.numpy()

plot_roc_curve(y_true_np, y_pred_np)
plot_precision_recall_curve(y_true_np, y_pred_np)
