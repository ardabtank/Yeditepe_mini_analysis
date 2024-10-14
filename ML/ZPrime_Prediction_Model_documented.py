'''
Importing necessary libraries and modules for data processing,
model creation, and evaluation which are important for any machine learning project.
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import pickle
import matplotlib.pyplot as plt
from tabulate import tabulate
from IPython.display import display, HTML
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, matthews_corrcoef, log_loss, roc_curve
import os
import plotly.graph_objects as go
from tensorflow.keras.layers import LeakyReLU
import json
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import RobustScaler


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
analyzed and collected from ATLAS 13 TeV Open Data. The dataset of "lep2" set's
Monte Carlo Simulations are analyzed and normalized with the corresponding
real world Data coming from ATLAS experiment.
'''

class DataProcessor:
    def __init__(self, file_paths, cache_file='cached_data.pkl'):
        self.file_paths = file_paths
        self.df = None
        self.scaler = StandardScaler()
        self.cache_file = cache_file

    '''
    Since the dataset we are working with is quite large, we save it as a cache in a pickle file.
    Caching is used to speed up the loading process. The pickle file allows us to store the data more
    efficiently compared to reading it directly from the CSV file each time. By doing this, we avoid
    reprocessing the data with each run, resulting in significantly faster load times.
    '''
    def load_data(self):
        if os.path.exists(self.cache_file):
            print(f"{self.cache_file} is found, loading from cache...")
            self.df = pd.read_pickle(self.cache_file)
        else:
            print("Cache is not found, loading from the CSV file...")
            dataframes = [pd.read_csv(file_path) for file_path in self.file_paths]
            self.df = pd.concat(dataframes, ignore_index=True)
            self.df.to_pickle(self.cache_file)
            print(f"Cached to the {self.cache_file} file.")

        '''
        We drop both the "DATA_MC_CHECK" column and "XSection" column. DATA_MC_CHECK represents the classification
        of signal and background: Signal is shown with "1" while the background is "0".
        and XSection represents the cross-section values of the Monte Carlo Simulations.
        We drop the 'XSection' column after extracting its values because these cross-section
        values will be used as "sample weights" during model training. In Monte Carlo simulations,
        the cross-section value represents the probability of a particular process occurring, and therefore
        provides crucial information about the relative importance of each event. By using cross-section values
        as sample weights, we ensure that events with higher cross-section (and thus higher likelihood of occurrence) are
        given more influence during training. This helps the model to better represent the real-world distribution of events
        observed in the experiment, preventing the model from being biased towards simulated events that might be over- or
        under-represented due to the nature of the simulation process.
        '''

        X = self.df.drop(['DATA_MC_CHECK', 'Xsection'], axis=1)
        y = self.df['DATA_MC_CHECK']

        '''
        Now we are going to calculate the weights with using both Xsection values and the "class weight" method. By multiplying each other,
        we both consider sample weights as cross section values and the class weights since we have an imbalanced dataset.
        And again, we save the weights as a pickle file.
        '''
        if os.path.exists('processed_data_with_weights.pkl'):
            print("Weights are loading from the pickle file...")
            self.df = pd.read_pickle('processed_data_with_weights.pkl')
            weights = self.df['weights']
        else:
            print("Weights are calculating...")
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
            class_weight_dict = dict(enumerate(class_weights))
            print("Class Weights:", class_weight_dict)
            #Multiplying both Xsection weights and class weights to get a final weight
            self.df['weights'] = self.df.apply(lambda row: row['Xsection'] * class_weight_dict[row['DATA_MC_CHECK']], axis=1)
            self.df.to_pickle('processed_data_with_weights.pkl')
            print("Weights are calculated and saved to a pickle file.")
            weights = self.df['weights']

        # Splitting the dataset into three parts: training, validation and test set (test set will be the unseen prediction set)
        X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
            X, y, weights, test_size=0.2, stratify=y, random_state=42)

        X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
            X_temp, y_temp, w_temp, test_size=0.0005, stratify=y_temp, random_state=42)
        '''
        Now by applying standard normalization with StandardScaler, (subtracting the mean and scaling to unit variance),
        we ensure that the input features for the training, validation, and test sets are on the
        same scale. This allows the model to process the data more efficiently and prevents it from
        focusing too heavily on features with larger numerical values, which are not necessarily more important
        physically.
        '''
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # Saving the scaler as a pickle to use it for later usings such as making predictions on the unseen test set.
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Creating a tf.data.Dataset from the test data and save it for later unseen-data prediction.
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test, w_test))
        tf.data.Dataset.save(test_dataset, "saved_test_dataset")
        print("Test dataset saved to 'saved_test_dataset' directory.")

        return (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test)

    '''
    We convert the dataset into a TensorFlow tf.data.Dataset object for efficient processing.
    Shuffling the dataset ensures that the model doesn’t learn patterns from the data order, while batching
    allows us to process large chunks of data at once, improving training speed and stability. Caching stores the
    preprocessed data in memory, reducing load times in subsequent epochs. Prefetching overlaps data loading with
    model training, maximizing resource utilization and reducing idle time. These optimizations help streamline
    the training process, especially with large datasets like ATLAS 13 TeV data.
    '''
    def preprocess_data(self, X, y, w):
        dataset = tf.data.Dataset.from_tensor_slices((X, y, w))

        BUFFER_SIZE = len(X)
        BATCH_SIZE = 65536
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        dataset = dataset.cache()

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


'''
Model is built using hyperparameters that were previously optimized through a hyperparameter
optimization process. The goal of this optimization was to find the best values for parameters
such as learning rate, regularization strength, and activation functions that maximize the model's
performance. By using these pre-optimized hyperparameters, we ensure that the model starts from a
strong configuration, avoiding the need for extensive manual tuning. The model architecture consists
of multiple fully connected layers with L2 regularization to prevent overfitting, and a LeakyReLU
activation in the input layer to avoid vanishing gradient issues. The output layer uses a sigmoid
activation for binary classification, making this model well-suited for distinguishing between signal and background events
'''

class ModelBuilder:
    def __init__(self):
        self.model = None

    def build_model(self):
        input_leaky_alpha = 0.11
        l2_input = 0.001222299763326385
        l2_reg = 3.158998297666698e-05
        learning_rate = 0.0033680902234094793

        # Building the model
        self.model = tf.keras.Sequential([

            #Input layer
            tf.keras.layers.Dense(24,
                                  kernel_regularizer=regularizers.l2(l2_input),
                                  input_shape=(12,)),
            tf.keras.layers.LeakyReLU(alpha=input_leaky_alpha),

            # First hidden layer
            tf.keras.layers.Dense(8,
                                  kernel_regularizer=regularizers.l2(l2_reg)),
            tf.keras.layers.ReLU(),

            # Second hidden layer
            tf.keras.layers.Dense(16,
                                  kernel_regularizer=regularizers.l2(l2_reg)),
            tf.keras.layers.ReLU(),

            # Third hidden layer
            tf.keras.layers.Dense(8,
                                  kernel_regularizer=regularizers.l2(l2_reg)),
            tf.keras.layers.ReLU(),

            # Output layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model with the given learning rate and metrics
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')],
            weighted_metrics=[
                tf.keras.metrics.BinaryAccuracy(name='weighted_accuracy'),
                tf.keras.metrics.AUC(name='weighted_auc'),
                tf.keras.metrics.Precision(name='weighted_precision'),
                tf.keras.metrics.Recall(name='weighted_recall')
            ]
        )

        return self.model

# Main using
if __name__ == "__main__":
    file_paths = ['Signal Dataset Here',
                  'Background Dataset Here']

    data_processor = DataProcessor(file_paths)

    (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test) = data_processor.load_data()

    train_dataset = data_processor.preprocess_data(X_train, y_train, w_train)
    validation_dataset = data_processor.preprocess_data(X_val, y_val, w_val)

    test_dataset = data_processor.preprocess_data(X_test, y_test, w_test)
    with open('test_dataset.pkl', 'wb') as f:
        pickle.dump((X_test, y_test, w_test), f)

    # Build and train the model
    model_builder = ModelBuilder()
    model = model_builder.build_model()

    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        epochs=100,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

    model.save('ZPrime_prediction_model.keras')
    model.summary()

    #Saving the model as config to use it for later purposes such as making predictions with unseen "test_data"
    model_config = model.get_config()
    json_path = "model_config.json"
    with open(json_path, 'w') as json_file:
        json.dump(model_config, json_file)

'''
Evaluation of the model with validation set is below.
Some performance metrics are shown with explanations and graphs are plotted.
Then the graphs are saved as png file.
'''

display(HTML(ColorPrinter.print_yellow("<br><br>EVALUATION OF THE MODEL:<br><br>")))


'''
We are evaluating the performance of the model based on both the training and 
validation datasets. The loss function provides a measure of how well the model 
is performing, with lower loss indicating better performance. In particular, we
are using binary cross-entropy loss since our task involves a binary classification 
problem—distinguishing signal from background events in the ATLAS 13 TeV dataset. Similarly, 
accuracy metrics show how well the model is classifying events correctly. By comparing the 
training and validation loss and accuracy, we can check if the model is overfitting (i.e., performing
well on training data but poorly on validation data) or if it is generalizing well.
The loss and accuracy are tracked over each epoch of training, and we plot the results 
to visualize the model's performance. A good model should ideally show a decreasing trend in
loss and increasing accuracy, while the gap between training and validation performance should remain
small, indicating a good generalization.
'''

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

validation_dataset_X = tf.concat([x for x, _, __ in validation_dataset], axis=0)
validation_dataset_y = tf.concat([y for _, y, __ in validation_dataset], axis=0)
validation_dataset_w = tf.concat([w for _, __, w in validation_dataset], axis=0)


'''
After training, we evaluate the network's ability to separate signal from
background events by examining the neural network output for both. We plot
histograms of the output for signal and background to visualize how well the
model distinguishes between the two. Ideally, we want to see clear separation 
between the signal and background distributions, where signal events cluster 
around one value and background events around another. Poor separation would indicate
that the model struggles to differentiate between the two, which could be a sign of issues
in the training process or model architecture.
'''

# Prediction with X
y_pred = model.predict(validation_dataset_X, batch_size=512)

# Taking predicted classes
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

'''
The classification report provides a breakdown of key metrics such as precision, recall, 
F1-score, and support for both signal and background classes. In the context of particle physics, 
high precision for the signal class is vital, as we want to ensure that when we predict a signal, 
it is very likely to be a true signal. Similarly, high recall is also important because missing 
true signal events could result in the loss of critical scientific discovery opportunities.
'''

# Evaluating the model on the validation dataset
eval_results = model.evaluate(validation_dataset)
print(f"Validation Metrics: {eval_results}")

# Generate classification report
report_dict = classification_report(validation_dataset_y, y_pred_classes, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
# Display as a table
display(report_df.style.background_gradient(cmap='viridis'))

'''
The Receiver Operating Characteristic (ROC) curve is an important tool in particle physics 
for evaluating the trade-off between true positive rate (signal efficiency) and false positive 
rate (background misclassification). In high-energy physics, it's crucial to balance these factors 
to detect rare signals without falsely identifying too many background events as signal. 
The Area Under the Curve (AUC) of the ROC curve gives a single metric of how well the model performs—an AUC 
closer to 1 indicates excellent classification performance.
'''

# ROC Curve drawing
def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)  # scikit-learn'ün roc_curve fonksiyonu
    plt.plot(fpr, tpr, linestyle='--', label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

'''
The precision-recall curve is particularly useful when dealing with 
imbalanced datasets, like the one we have in particle physics where 
background events vastly outnumber signal events. Precision tells us how many 
of the predicted signal events are actually true positives, while recall shows 
how many of the actual signal events were correctly identified. By plotting this curve, 
we can understand the trade-off between these two metrics and ensure the model is making 
reliable predictions for signal events.
'''

# Precision-Recall Curve drawing
def plot_precision_recall_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)  # scikit-learn'ün precision_recall_curve fonksiyonu
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.show()

# TensorFlow tensors to numpy to plot ROC and Precision-Recall graphs
y_true_np = validation_dataset_y.numpy()
y_pred_np = y_pred.numpy()

# Plotting ROC ve Precision-Recall graphs
plot_roc_curve(y_true_np, y_pred_np)
plot_precision_recall_curve(y_true_np, y_pred_np)
