import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Параметры, которые нужны для обнаружения движения
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Цветовые палитры, которые используются  для хитмапа
colormaps = {
  'viridis': plt.cm.viridis,
  'jet': plt.cm.jet,
  'hot': plt.cm.hot,
  'coolwarm': plt.cm.coolwarm
}

#Aункция, которая запускается, когда пользователь нажимает кнопку "Загрузить видео" в интерфейсе
def load_video():
  global cap, heatmap
  filepath = filedialog.askopenfilename(
    defaultextension=".mp4",
    filetypes=(("Video files", "*.mp4"), ("All files", "*.*"))
  )
  if filepath:
    cap = cv2.VideoCapture(filepath)
    heatmap = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.uint8)
    process_video()

#Функция, которая обрабатывает выбранное видео
def process_video():
  global cap, heatmap
  heatmap = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.uint8)

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      if cv2.contourArea(contour) > 100:
        x, y, w, h = cv2.boundingRect(contour)
        heatmap[y:y+h, x:x+w] += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Рисует прямоугольник

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

  # Визуализация хитмапа
  heatmap = cv2.normalize(heatmap, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
  fig, ax = plt.subplots()
  image = ax.imshow(heatmap, cmap=colormaps['viridis']) # Используем "viridis" по умолчанию
  canvas = FigureCanvasTkAgg(fig, master=root)
  canvas_widget = canvas.get_tk_widget()
  canvas_widget.pack(fill=tk.BOTH, expand=True)

# Создание GUI
root = tk.Tk()
root.title("Хитмап видео")

# Кнопка "Загрузить видео"
load_button = tk.Button(root, text="Загрузить видео", command=load_video)
load_button.pack()

root.mainloop()
