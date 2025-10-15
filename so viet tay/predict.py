import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

class DrawApp:
    def __init__(self, model_path='mnist_cnn.h5'):
        self.model = load_model(model_path)

        self.root = tk.Tk()
        self.root.title("Nhận diện chữ số viết tay")

        self.canvas_width = 280
        self.canvas_height = 280

        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(self.root, text="Dự đoán", command=self.predict)
        self.button_predict.pack()

        self.button_clear = tk.Button(self.root, text="Xóa", command=self.clear)
        self.button_clear.pack()

        self.label_result = tk.Label(self.root, text="Kết quả: ", font=("Helvetica", 16))
        self.label_result.pack()

        # Tạo ảnh trắng để vẽ
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image1)

        # Bắt sự kiện vẽ chuột
        self.canvas.bind("<B1-Motion>", self.paint)

        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill='white')
        self.label_result.config(text="Kết quả: ")

    def preprocess_image(self):
        # Resize ảnh vẽ về 28x28, đảo màu (vì MNIST nền đen chữ trắng)
        img = self.image1.resize((28,28))
        img = ImageOps.invert(img)
        img = np.array(img).astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # (28,28,1)
        img = np.expand_dims(img, axis=0)   # (1,28,28,1)
        return img

    def predict(self):
        img = self.preprocess_image()
        preds = self.model.predict(img)
        predicted_class = np.argmax(preds)
        confidence = preds[0][predicted_class]
        self.label_result.config(text=f"Kết quả: {predicted_class} (xác suất {confidence*100:.2f}%)")

if __name__ == '__main__':
    DrawApp()
