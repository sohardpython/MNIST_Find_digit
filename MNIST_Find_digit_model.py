# 모델 생성
def create_cnn(learning_rate):
    # Remove the previous model.
    model = None

    # Input layer
    img_input = layers.Input(shape=(28, 28, 1))

    # CNN
    # Identity mapping shortcut을 위한 conv_1 layer
    conv_1 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(img_input)

    conv_2_1 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv_1)
    conv_2_1 = layers.Conv2D(128, kernel_size=3, padding='same')(conv_2_1)

    # ShortCut connection
    add_2_1 = layers.add([conv_1, conv_2_1])
    out_2_1 = layers.Activation('relu')(add_2_1)

    conv_2_2 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(out_2_1)
    conv_2_2 = layers.Conv2D(128, kernel_size=3, padding='same')(conv_2_2)

    # ShortCut connection
    add_2_2 = layers.add([out_2_1, conv_2_2])
    out_2_2 = layers.Activation('relu')(add_2_2)

    pool_2 = layers.MaxPool2D((2, 2), strides=2)(out_2_2)

    conv_3_0 = layers.Conv2D(256, kernel_size=1, strides=1)(pool_2)

    conv_3_1 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv_3_0)
    conv_3_1 = layers.Conv2D(256, kernel_size=3, padding='same')(conv_3_1)

    # ShortCut connection
    add_3_1 = layers.add([conv_3_0, conv_3_1])
    out_3_1 = layers.Activation('relu')(add_3_1)

    conv_3_2 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(out_3_1)
    conv_3_2 = layers.Conv2D(256, kernel_size=3, padding='same')(conv_3_2)

    # ShortCut connection
    add_3_2 = layers.add([out_3_1, conv_3_2])
    out_3_2 = layers.Activation('relu')(add_3_2)

    pool_3 = layers.MaxPool2D((2, 2), strides=2)(out_3_2)

    conv_4_0 = layers.Conv2D(256, kernel_size=1, strides=1)(pool_3)

    conv_4_1 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv_4_0)
    conv_4_1 = layers.Conv2D(256, kernel_size=3, padding='same')(conv_4_1)

    # ShortCut connection
    add_4_1 = layers.add([conv_4_0, conv_4_1])
    out_4_1 = layers.Activation('relu')(add_4_1)

    pool_4 = layers.MaxPool2D((2, 2), strides=2)(out_4_1)

    # FC layers
    img_features = layers.Flatten()(pool_4)
    img_features = layers.Dense(512, activation='relu')(img_features)
    img_features = layers.Dropout(rate=0.5)(img_features)
    img_features = layers.Dense(512, activation='relu')(img_features)
    img_features = layers.Dropout(rate=0.5)(img_features)

    # Output layer
    digit_pred = layers.Dense(10, activation='softmax')(img_features)

    model = keras.Model(inputs=img_input, outputs=digit_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

model = create_cnn(learning_rate = 0.0001)
model.summary()
del model


# 모델사용을 위해 함수로 지정
def train_model_v1(model, X_train, y_train, X_val, y_val, epochs, batch_size=None, validation_split=0.1):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        shuffle=True, validation_data=(X_val, y_val), callbacks=[callback])

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


def train_model_v2(model, X_train, y_train, X_val, y_val, epochs, batch_size=None, validation_split=0.1):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        shuffle=True, validation_data=(X_val, y_val), callbacks=[callback])

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

# Ensemble을 위해 list에 저장
model_list = []
for i in range(10):
    model = create_cnn(learning_rate = 0.0001)
    model_list.append(model)

# Training
epochs = 200
batch_size = 16
validation_split = 0.2

for i in range(len(model_list)):
    print("***************Trainig_my_model_{}*****************".format(i))
    epoch, hist = train_model_v1(model_list[i], X_train, y_train, X_val, y_val, epochs, batch_size)


# 모델들의 accuracy 보기
result_list = []
for i in range(len(model_list)):
    print("************************Evaluating_my_model_{}************************".format(i))
    result = model_list[i].evaluate(X_val, y_val)
    result_list.append(result)

# test_set predict
pred_list = []
for i in range(len(model_list)):
    print("************************Predicting_my_model_{}************************".format(i))
    pred = model_list[i].predict(X_test)
    pred_list.append(pred)

# model들의 test데이터 정답예측 dataframe으로 확인
pred_df = pd.DataFrame(test["id"])
final_pred = np.array([0] * 204800).reshape(20480, 10)

for i in range(len(model_list)):
    pred_df['pred{}'.format(i)] = np.argmax(pred_list[i], axis = 1)
    final_pred = final_pred + pred_list[i]

pred_df['final_pred'] = np.argmax(final_pred, axis = 1)

# train_set으로 학습이 끝난 후 val_set으로 다시 학습
re_model_list = []
for i in range(len(model_list)):
    print("************************Loading_my_model_{}************************".format(i))

    my_model = model_list[i]
    my_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

    print("************************Re-training_my_model_{}************************".format(i))
    my_model.fit(X_val, y_val, epochs=5, batch_size=1)

    re_model_list.append(my_model)

# test_set 다시 predict
re_pred_list = []
for i in range(len(model_list)):
    print("************************Predicting_my_re_model_{}************************".format(i))
    pred = re_model_list[i].predict(X_test)
    re_pred_list.append(pred)

re_pred_df = pd.DataFrame(test["id"])
re_final_pred = np.array([0] * 204800).reshape(20480, 10)

# 다시 model들의 test데이터 정답예측 dataframe으로 확인
for i in range(len(model_list)):
    re_pred_df['pred{}'.format(i)] = np.argmax(re_pred_list[i], axis = 1)
    re_final_pred = re_final_pred + re_pred_list[i]

re_pred_df['final_pred'] = np.argmax(re_final_pred, axis = 1)