from data_preprocessing import load_and_preprocess_data
from model import build_cnn_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_cnn_model()

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Lưu mô hình tốt nhất dựa trên accuracy trên tập validation
    checkpoint = ModelCheckpoint('mnist_cnn.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint])

    # Đánh giá trên tập test
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc*100:.2f}%")

if __name__ == '__main__':
    train_model()
