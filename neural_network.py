#
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.regularizers import l2
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np
# from keras.optimizers import Adam
# from sklearn.model_selection import ParameterGrid
# import os
#
#
# def build_and_train_nn(X_train, Y_train, X_test, Y_test, save_path="best_nn_model.h5", use_grid_search=False):
#     """
#     Builds and trains a neural network model with hyperparameter tuning.
#
#     Parameters:
#         X_train: Training feature set.
#         Y_train: Training target set.
#         X_test: Testing feature set.
#         Y_test: Testing target set.
#         save_path: File path to save the trained model.
#         use_grid_search: If True, performs a basic grid search for hyperparameter tuning.
#
#     Returns:
#         model: Trained Keras model.
#         history: Training history object.
#     """
#
#     def create_model(dropout_rate, learning_rate, l2_reg):
#         model = Sequential([
#             Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(l2_reg)),
#             BatchNormalization(),
#             Dropout(dropout_rate),
#             Dense(64, activation='relu'),
#             BatchNormalization(),
#             Dropout(dropout_rate),
#             Dense(32, activation='relu'),
#             BatchNormalization(),
#             Dropout(dropout_rate),
#             Dense(1, activation='sigmoid')
#         ])
#         model.compile(
#             optimizer=Adam(learning_rate=learning_rate),
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
#         return model
#
#     # Compute class weights to handle class imbalance
#     class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
#     class_weights = dict(enumerate(class_weights))
#
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
#     model_checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7)
#
#     if use_grid_search:
#         param_grid = {
#             'dropout_rate': [0.3, 0.4, 0.5],
#             'learning_rate': [0.0005, 0.0001, 0.00005],
#             'l2_reg': [0.001, 0.0005, 0.0001]
#         }
#         grid = ParameterGrid(param_grid)
#         best_val_loss = float('inf')
#         best_model = None
#         best_history = None
#         best_params = None
#
#         for params in grid:
#             print(f"Training with params: {params}")
#             model = create_model(params['dropout_rate'], params['learning_rate'], params['l2_reg'])
#             history = model.fit(
#                 X_train, Y_train,
#                 validation_data=(X_test, Y_test),
#                 epochs=50,
#                 batch_size=16,
#                 callbacks=[early_stopping, reduce_lr],
#                 class_weight=class_weights,
#                 verbose=0
#             )
#
#             val_loss = min(history.history['val_loss'])
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model = model
#                 best_history = history
#                 best_params = params
#             print(f"Current best val_loss: {best_val_loss}")
#
#         print(f"Best parameters: {best_params}")
#
#         model_checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
#         best_history = best_model.fit(
#             X_train, Y_train,
#             validation_data=(X_test, Y_test),
#             epochs=50,
#             batch_size=16,
#             callbacks=[early_stopping, model_checkpoint, reduce_lr],
#             class_weight=class_weights,
#             verbose=1
#         )
#         print(f"Best Neural Network model saved to {save_path}.")
#         return best_model, best_history
#     else:
#         model = create_model(dropout_rate=0.4, learning_rate=0.0001, l2_reg=0.0005)
#
#         history = model.fit(
#             X_train, Y_train,
#             validation_data=(X_test, Y_test),
#             epochs=50,
#             batch_size=16,
#             callbacks=[early_stopping, model_checkpoint, reduce_lr],
#             class_weight=class_weights,
#             verbose=1
#         )
#         print(f"Best Neural Network m   odel saved to {save_path}.")
#         return model, history
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras.callbacks import LearningRateScheduler

def custom_lr_schedule(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(custom_lr_schedule)

def build_and_train_nn(X_train, Y_train, X_test, Y_test, save_path="best_nn_model.h5"):
    """
    Builds and trains a neural network model with regularization and learning rate adjustment.

    Parameters:
        X_train: Training feature set.
        Y_train: Training target set.
        X_test: Testing feature set.
        Y_test: Testing target set.
        save_path: File path to save the trained model.

    Returns:
        model: Trained Keras model.
        history: Training history object.
    """
    model = Sequential([
        Dense(512, activation='relu', input_dim=X_train.shape[1],kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    from keras.optimizers import RMSprop
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights = dict(enumerate(class_weights))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

    # Model training
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=30,
        batch_size=128,  # Adjusted batch size
        callbacks=[early_stopping, model_checkpoint, reduce_lr, lr_scheduler],
        class_weight=class_weights
    )

    print(f"Best Neural Network model saved to {save_path}.")
    return model, history