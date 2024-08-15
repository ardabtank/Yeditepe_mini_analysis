'''
Importing necessary libraries and modules for data processing, 
model creation and evaluation which are essential for a deep learning project
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras import regularizers # type: ignore
import keras_tuner as kt
import pickle
import matplotlib.pyplot as plt
from tabulate import tabulate
from IPython.display import display, HTML
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, matthews_corrcoef, log_loss, roc_curve
import os
import plotly.graph_objects as go #type: ignore
from imblearn.under_sampling import RandomUnderSampler  # type: ignore
from tensorflow.keras.layers import LeakyReLU # type: ignore

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
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.df = None
        self.df_balanced = None
        self.color_printer = ColorPrinter()
    
    #Loading the relevant files which including our datasets.
    def load_data(self): 
        dataframes = []
        for file_path in self.file_paths:
            df = pd.read_csv(file_path)
            dataframes.append(df)

        self.df = pd.concat(dataframes, ignore_index=True)
        return self.df

    
    def balance_data(self):
        X = self.df.drop('DATA_MC_CHECK', axis=1)
        y = self.df['DATA_MC_CHECK']

        '''
        The signal and background datasets converted to DataFrame and the
        non-feature column is dropped above. 

        Now they need to be balanced depending on their # of rows. 
        It'll be done with RandomUnderSample (from imbalanced-learn library) 
        which detects minority/majority sets automatically and under-sample 
        the majority class(es) by randomly picking without replacement. 

        Then the relevant minority and majority classes combined into one set.
        '''
        
        under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = under_sampler.fit_resample(X, y)

        self.df_balanced = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='DATA_MC_CHECK')], axis=1)

        # Saving the number of rows and a sample.
        with open('checking.txt', 'w') as file:
            file.write(f"Total number of rows in balanced dataset: {len(self.df_balanced)}\n\n")
            file.write(f"Number of rows in minority class: {self.df_balanced['DATA_MC_CHECK'].value_counts().min()}\n\n")
            file.write(f"Number of rows in majority class: {self.df_balanced['DATA_MC_CHECK'].value_counts().max()}\n\n")
            file.write("Sample of 30 rows: \n")
            file.write(self.df_balanced.sample(n=30, random_state=42).to_string(index=False))
        return self.df_balanced


    '''
    Splitting the balanced dataset into three parts: training, validation
    and test parts. While doing it, considering the weights which are the
    XSection values is important. XSection values are taken as a weight since 
    those values represent the probability of emerging an event in Particle Physics experiments.
    '''
    def preprocess_data(self):
      self.df_balanced['weight'] = self.df_balanced['Xsection']
      X = self.df_balanced.drop(['DATA_MC_CHECK', 'Xsection'], axis=1)
      y = self.df_balanced['DATA_MC_CHECK']
      weights = self.df_balanced['weight']


      # First split the dataset into 80% training and 20% temporary set
      skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      train_index, temp_index = next(skf.split(X, y))
      X_train, X_temp = X.iloc[train_index], X.iloc[temp_index]
      y_train, y_temp = y.iloc[train_index], y.iloc[temp_index]
      w_train, w_temp = weights.iloc[train_index], weights.iloc[temp_index]

      # Further split temp into 75% validation and 25% test sets
      skf_temp = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
      val_index, test_index = next(skf_temp.split(X_temp, y_temp))
      X_val, X_test = X_temp.iloc[val_index], X_temp.iloc[test_index]
      y_val, y_test = y_temp.iloc[val_index], y_temp.iloc[test_index]
      w_val, w_test = w_temp.iloc[val_index], w_temp.iloc[test_index]
      
      '''
      Finally we have %80 training, %15 validation and %5 test set.
      '''

      # Normalization of datasets
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_val = scaler.transform(X_val)
      X_test = scaler.transform(X_test)
      
      '''
      X_train is scaled by its own mean and standart deviation.
      X_val and X_test is scaled by the same mean and standart deviation
      values as X_train. This is because we need to ensure that the model
      evaluates new and unseen data (X_test) under the same conditions as
      it was trained.
      '''

      return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test 
    
    
    #Checking if there is any intersections between training, validation and test sets
    def check_intersections(self, X_train, X_val, X_test):
      
      X_train_df = pd.DataFrame(X_train)
      X_val_df = pd.DataFrame(X_val)
      X_test_df = pd.DataFrame(X_test)

      train_val_intersection = pd.merge(X_train_df, X_val_df, how='inner')
      train_test_intersection = pd.merge(X_train_df, X_test_df, how='inner')
      val_test_intersection = pd.merge(X_val_df, X_test_df, how='inner')

      print(f"Train-Val Intersection: {len(train_val_intersection)} rows")
      print(f"Train-Test Intersection: {len(train_test_intersection)} rows")
      print(f"Val-Test Intersection: {len(val_test_intersection)} rows")

      return train_val_intersection, train_test_intersection, val_test_intersection

'''
Class HyperModel has the hyperparameter optimization. It tries different combinations
of hyperparameters to find the best accuracy for predictions.
'''
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential()

        # Fixed input layer with 12 neurons and try relu and leaky_relu as activation functions.
        input_activation = hp.Choice('input_activation', values=['relu', 'leaky_relu'], default='relu')
        model.add(tf.keras.layers.Dense(
            12,
            kernel_regularizer=regularizers.l2(hp.Float('l2_input', 1e-5, 1e-2, sampling='log'))
        ))

        # Adding activation for input layer based on hyperparameter choice
        if input_activation == 'leaky_relu':
            # Tune alpha for Leaky ReLU
            alpha_input = hp.Float('leaky_alpha_input', 0.01, 0.3, step=0.05, default=0.01)
            model.add(LeakyReLU(alpha=alpha_input))
        else:
            model.add(tf.keras.layers.Activation('relu'))

        # Hidden layers with tunable configurations
        for i in range(hp.Int('num_hidden_layers', 1, 4)):
            units = hp.Int(f'units_{i}', min_value=8, max_value=32, step=8)
            activation = hp.Choice(f'activation_{i}', values=['relu', 'leaky_relu'], default='relu')
            l2_reg = hp.Float(f'l2_{i}', 1e-5, 1e-2, sampling='log')

            model.add(tf.keras.layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg)))

            if activation == 'leaky_relu':
                alpha_hidden = hp.Float(f'leaky_alpha_{i}', 0.01, 0.3, step=0.05, default=0.01)
                model.add(LeakyReLU(alpha=alpha_hidden))
            else:
                model.add(tf.keras.layers.Activation('relu'))

        # Output layer with sigmoid activation for binary classification
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Compile the model with a tunable learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    '''
    Below method trains a Keras model using a custom training loop with hyperparameter tuning for batch size and learning rate. 
    It converts training and validation data into `tf.data.Dataset` objects for efficient batching. 
    The method uses the Adam optimizer and binary cross-entropy loss, tracking performance with mean loss 
    and binary accuracy metrics. 

    Includes analyzing the performance metrics such as epoch_loss_metric and epoch_accuracy_metric and determining the best
    epoch accuracy.
    '''
    def fit(self, hp, model, X_train, y_train, validation_data, sample_weight=None, callbacks=None, **kwargs):
        batch_size = hp.Int('batch_size', 128, 512, step=128)
        if callbacks is None:
            callbacks = []
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weight)).batch(batch_size)
        X_val, y_val, w_val = validation_data
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val, w_val)).batch(batch_size)

        optimizer = tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        epoch_loss_metric = tf.keras.metrics.Mean()
        epoch_accuracy_metric = tf.keras.metrics.BinaryAccuracy()

        @tf.function
        def run_train_step(images, labels, sample_weight):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits, sample_weight=sample_weight)
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_accuracy_metric.update_state(labels, logits, sample_weight=sample_weight)

        @tf.function
        def run_val_step(images, labels, sample_weight):
            logits = model(images)
            loss = loss_fn(labels, logits, sample_weight=sample_weight)
            epoch_loss_metric.update_state(loss)
            epoch_accuracy_metric.update_state(labels, logits, sample_weight=sample_weight)

        for callback in callbacks:
            callback.set_model(model)

        best_epoch_accuracy = 0.0

        for epoch in range(4):  # Set to a small number for validating, increase as needed
            print(f"Epoch: {epoch}")

            for images, labels, sample_weight in train_ds:
                run_train_step(images, labels, sample_weight)

            for images, labels, sample_weight in val_ds:
                run_val_step(images, labels, sample_weight)

            epoch_loss = float(epoch_loss_metric.result().numpy())
            epoch_accuracy = float(epoch_accuracy_metric.result().numpy())
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={'val_loss': epoch_loss, 'val_accuracy': epoch_accuracy})
            epoch_loss_metric.reset_state()
            epoch_accuracy_metric.reset_state()

            print(f"Epoch loss: {epoch_loss}, Epoch accuracy: {epoch_accuracy}")
            best_epoch_accuracy = max(best_epoch_accuracy, epoch_accuracy)

        return best_epoch_accuracy



if __name__ == "__main__":
    # Load and preprocess data
    file_paths = ['ZPrime.csv', 'background.csv']
    data_processor = DataProcessor(file_paths)
    data_processor.load_data()
    data_processor.balance_data()
    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test = data_processor.preprocess_data()
    data_processor.check_intersections(X_train, X_val, X_test)

    hypermodel = MyHyperModel()

    tuner = kt.Hyperband(
        hypermodel,
        objective=kt.Objective('val_accuracy', 'max'),  
        max_epochs=100,
        factor=3,
        directory='kerastunerX',
        project_name='keras_tuner_project',
        overwrite=True
    )
    '''
    Tuning process aims to maximize validation accuracy. Below part
    conducts a hyperparameter search on the training data with early
    stopping based on validation loss. 
    '''
    tuner.search(X_train, y_train, validation_data=(X_val, y_val, w_val), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=10)])

    '''
    The search identifies the best hyperparameters for the model.
    '''
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    best_model = tuner.get_best_models(num_models=1)[0]
    history = best_model.fit(X_train, y_train,
                   sample_weight=w_train,
                    epochs=100,
                    batch_size=best_hps.get('batch_size'),
                    validation_data=(X_val, y_val, w_val),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    '''
    Training the model with the best hyperparameters on the training
    data. The model is validated on the validation set. Finally, saving
    below.
    '''


    best_model.save('ZPrime_prediction_model.keras')


'''
Evaluation of the model with validation set is below.

Some performance metrics are shown with explanations and graphs are plotted. 
Then the graphs are saved as png file into a folder named Graphs.
'''

display(HTML(ColorPrinter.print_yellow("<br><br>EVALUATION OF THE MODEL:<br><br>")))

y_pred = best_model.predict(X_val)
y_pred_classes = (y_pred > 0.5).astype("int32")

accuracy = accuracy_score(y_val, y_pred_classes, sample_weight=w_val)
f1 = f1_score(y_val, y_pred_classes, sample_weight=w_val)
roc_auc = roc_auc_score(y_val, y_pred, sample_weight=w_val)
precision = precision_score(y_val, y_pred_classes, sample_weight=w_val)
recall = recall_score(y_val, y_pred_classes, sample_weight=w_val)
mcc = matthews_corrcoef(y_val, y_pred_classes, sample_weight=w_val)
logloss = log_loss(y_val, y_pred, sample_weight=w_val)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Precision:", precision)
print("Recall:", recall)
print("MCC:", mcc)
print("Log Loss:", logloss)


def evaluate_metric(metric_name, value):
    if metric_name in ["Accuracy", "F1 Score", "Precision", "Recall"]:
        if value < 0.60:
            return "Bad"
        elif value < 0.75:
            return "Good enough"
        elif value < 0.85:
            return "Good"
        elif value < 0.90:
            return "Very good"
        else:
            return "Extremely good"
    elif metric_name == "ROC AUC Score":
        if value < 0.65:
            return "Bad"
        elif value < 0.75:
            return "Good enough"
        elif value < 0.85:
            return "Good"
        elif value < 0.90:
            return "Very good"
        else:
            return "Extremely good"
    elif metric_name == "Log Loss":
        if value > 1.0:
            return "Bad"
        elif value > 0.5:
            return "Good enough"
        elif value > 0.2:
            return "Good"
        elif value > 0.1:
            return "Very good"
        else:
            return "Extremely good"
    elif metric_name == "Matthews Correlation Coefficient":
        if value < 0.0:
            return "Bad"
        elif value < 0.30:
            return "Good enough"
        elif value < 0.60:
            return "Good"
        elif value < 0.85:
            return "Very good"
        else:
            return "Extremely good"
    return "Unknown"

# THE PROCESS OF PERFORMANCE METRICS
display(HTML(ColorPrinter.print_yellow("<br><br>PERFORMANCE METRICS:<br><br>")))

metrics = {
    "Accuracy": accuracy,
    "F1 Score": f1,
    "ROC AUC Score": roc_auc,
    "Precision": precision,
    "Recall": recall,
    "Matthews Correlation Coefficient": mcc,
    "Log Loss": logloss
}

for metric_name, value in metrics.items():
    evaluation = evaluate_metric(metric_name, value)
    if evaluation == "Bad":
        display(HTML(f"{metric_name}: {value:.10f} - {ColorPrinter.print_red(evaluation)}"))
    elif evaluation == "Good enough":
        display(HTML(f"{metric_name}: {value:.10f} - {ColorPrinter.print_magenta(evaluation)}"))
    elif evaluation == "Good":
        display(HTML(f"{metric_name}: {value:.10f} - {ColorPrinter.print_cyan(evaluation)}"))
    elif evaluation == "Very good":
        display(HTML(f"{metric_name}: {value:.10f} - {ColorPrinter.print_yellow(evaluation)}"))
    elif evaluation == "Extremely good":
        display(HTML(f"{metric_name}: {value:.10f} - {ColorPrinter.print_green(evaluation)}"))

# Explanation of metrics above.
display(HTML(ColorPrinter.print_yellow("<br>Explanations of above metrics:<br>")))
display(HTML("<br><strong>Accuracy:</strong> Measures the overall correctness of the model's predictions. It is the ratio of correctly predicted instances to the total instances. High accuracy in particle physics ensures reliable identification of particle events."))
display(HTML("<br><strong>F1 Score:</strong> The harmonic mean of precision and recall. It balances the trade-off between precision and recall, making it a useful metric for evaluating the performance of classifiers, especially on imbalanced datasets commonly found in particle physics."))
display(HTML("<br><strong>ROC AUC Score:</strong> Represents the area under the Receiver Operating Characteristic curve. A higher AUC indicates better model performance in distinguishing between the positive and negative classes. In particle physics, a high AUC score is crucial for distinguishing between signal and background events."))
display(HTML("<br><strong>Precision:</strong> The ratio of true positive predictions to the total predicted positives. High precision means that the classifier has a low false positive rate, which is vital in particle physics to ensure that identified particle events are actually signal events."))
display(HTML("<br><strong>Recall:</strong> The ratio of true positive predictions to the actual positives. High recall ensures that most of the actual signal events are identified by the model, which is critical in experiments where missing a signal event can be costly."))
display(HTML("<br><strong>Matthews Correlation Coefficient:</strong> A measure of the quality of binary classifications. It takes into account true and false positives and negatives, and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. High MCC indicates a strong correlation between the predicted and actual classes, which is essential in particle physics for reliable event classification."))
display(HTML("<br><strong>Log Loss:</strong> Measures the performance of a classification model where the prediction input is a probability value between 0 and 1. Lower log loss indicates better performance. In particle physics, minimizing log loss helps in achieving more accurate probabilistic predictions for particle events, improving overall model reliability."))


display(HTML(ColorPrinter.print_yellow("<br><br>GRAPHS:<br>")))

# DEFINING THE LOSS OF TRAIN AND TEST
loss = history.history['loss']
val_loss = history.history['val_loss']

# DEFINING THE ACCURACY OF TRAIN AND TEST
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)

# THE GRAPH OF TRAINING AND VALIDATION LOSS
display(HTML(ColorPrinter.print_orange("<br><br>TRAINING AND VALIDATION LOSS:<br><br>")))
print("The Training and Validation Loss graphs illustrate the model's performance over the training epochs. The loss function quantifies the difference between the predicted and true values. In particle physics experiments, minimizing the loss function is crucial for improving the accuracy of particle identification models. A decreasing training loss indicates that the model is learning from the data, while the validation loss helps in monitoring overfitting.\n")
plt.figure(figsize=(12, 5))
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

output_path = os.path.join('Graphs', "training_and_validation_loss.png")
plt.savefig(output_path)

plt.show()

# THE GRAPH OF TRAINING AND VALIDATION ACCURACY
display(HTML(ColorPrinter.print_orange("<br><br>TRAINING AND VALIDATION ACCURACY:<br><br>")))
print("The Training and Validation Accuracy graphs show how well the model's predictions match the actual data over the epochs. High accuracy means the model is correctly identifying signal and background events. In the context of particle physics, high accuracy ensures that we can trust the model's predictions when identifying rare particle interactions or decays.\n")
plt.figure(figsize=(12, 5))
plt.plot(epochs, train_accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
output_path1 = os.path.join('Graphs', "training_and_validation_accuracy.png")
plt.savefig(output_path1)
plt.show()

# Plotting ROC curve
def plot_roc_curve(y_val, y_pred, sample_weight):

  fpr, tpr, thresholds = roc_curve(y_val, y_pred, sample_weight=sample_weight)
  
  
  plt.plot(fpr, tpr, linestyle='--', label='ROC Curve')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend()
  output_dir = "Graphs"
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  output_path = os.path.join(output_dir, "roc_curve.png")
  plt.savefig(output_path)
  plt.show()

display(HTML(ColorPrinter.print_orange("<br><br>ROC CURVE:<br><br>")))
print("The ROC curve, or Receiver Operating Characteristic curve, is a graphical representation of a classifier's performance. It plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) at various threshold settings. In particle physics, this curve helps in understanding the trade-off between detecting true signal events and the rate of false alarms. A perfect classifier has an area under the ROC curve (AUC) of 1, indicating it perfectly distinguishes between signal and background. \n")
plot_roc_curve(y_val, y_pred, sample_weight=w_val)


# PRECISION-RECALL CURVE PLOT
display(HTML(ColorPrinter.print_orange("<br><br>PRECISION-RECALL CURVE:<br><br>")))
print("The Precision-Recall curve is crucial for evaluating the performance of classifiers on imbalanced datasets. Precision indicates the fraction of true positive predictions among all positive predictions, while recall measures the fraction of true positives identified among all actual positives. In particle physics, high precision and recall are vital for ensuring that rare particle events are correctly identified without being overwhelmed by false positives.\n")
precision, recall, _ = precision_recall_curve(y_val, y_pred)
plt.figure()
plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
output_path2 = os.path.join('Graphs', "precision_recall_curve.png")
plt.savefig(output_path2)
plt.show()




#Creating an example table with splitted X_test set

# Making predictions on the validation
test_predictions = best_model.predict(X_test)

# Converting prediction probabilities to binary predictions
prediction_classes_test = [1 if prob > 0.5 else 0 for prob in test_predictions]

# Creating a DataFrame to display the results
results_df = pd.DataFrame({
    'True Class': y_test,
    'Predicted Class': prediction_classes_test,
    'Prediction Probability': test_predictions.flatten()
})

# Creating the table
fig = go.Figure(data=[go.Table(
  header=dict(values=list(results_df.columns),
              fill_color='paleturquoise',
              align='left'),
  cells=dict(values=[results_df[col] for col in results_df.columns],
             fill_color='lavender',
             align='left'))
])

# Saving the table as HTML.
fig.write_html('prediction_on_test_results.html')

# Confusion Matrix of this prediction
cm = confusion_matrix(y_test, prediction_classes_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax)
plt.title('Confusion Matrix')
output_path_cm = os.path.join('Graphs', "confusion_matrix_onTestData.png")
plt.savefig(output_path_cm, bbox_inches='tight')
plt.show()


