import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
import cv2
from skimage.morphology import skeletonize
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")

        self.image_path = None
        self.img = None
        self.cropped_img = None
        self.gray = None
        self.skeleton = None
        self.pixels_to_microns = 1.0

        # Widgets
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.pixels_to_microns_label = tk.Label(root, text="Pixels to Microns Ratio")
        self.pixels_to_microns_label.grid(row=1, column=0, padx=5, pady=5)
        self.pixels_to_microns_entry = tk.Entry(root)
        self.pixels_to_microns_entry.insert(0, "200/190")
        self.pixels_to_microns_entry.grid(row=1, column=1, padx=5, pady=5)

        self.crop_label = tk.Label(root, text="Crop (Start Row, End Row, Start Column, End Column)")
        self.crop_label.grid(row=2, column=0, padx=5, pady=5)
        self.crop_entry = tk.Entry(root)
        self.crop_entry.insert(0, "0,800,0,500")
        self.crop_entry.grid(row=2, column=1, padx=5, pady=5)

        self.edges_label = tk.Label(root, text="Canny Thresholds (low, high)")
        self.edges_label.grid(row=3, column=0, padx=5, pady=5)
        self.edges_entry = tk.Entry(root)
        self.edges_entry.insert(0, "20,100")
        self.edges_entry.grid(row=3, column=1, padx=5, pady=5)

        self.show_cut_button = tk.Button(root, text="Show Cut Image", command=self.show_cut_image)
        self.show_cut_button.grid(row=4, column=0, padx=5, pady=5)

        self.show_skeleton_button = tk.Button(root, text="Show Skeleton", command=self.show_skeleton)
        self.show_skeleton_button.grid(row=4, column=1, padx=5, pady=5)

        self.calculate_length_button = tk.Button(root, text="Calculate Length", command=self.calculate_length)
        self.calculate_length_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        self.result_label = tk.Label(root, text="Skeleton Length: N/A")
        self.result_label.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

        self.canvas = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.img = cv2.imread(self.image_path)
            self.show_image(self.img, "Original Image")

    def crop_image(self):
        if self.img is None:
            return
        try:
            crop_values = list(map(int, self.crop_entry.get().split(",")))
            start_row, end_row, start_col, end_col = crop_values
            self.cropped_img = self.img[start_row:end_row, start_col:end_col]
        except Exception as e:
            tk.messagebox.showerror("Error", f"Invalid crop values: {e}")

    def process_image(self):
        if self.cropped_img is None:
            return

        self.gray = cv2.cvtColor(self.cropped_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(self.gray)
        blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

        edges = cv2.Canny(blurred, *map(int, self.edges_entry.get().split(",")))
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.skeleton = skeletonize(closed > 0)

    def calculate_length(self):
        self.crop_image()
        self.process_image()

        if self.skeleton is not None:
            skeleton_pixels = np.sum(self.skeleton)
            self.pixels_to_microns = eval(self.pixels_to_microns_entry.get())
            length_microns = skeleton_pixels * self.pixels_to_microns
            self.result_label.config(text=f"Skeleton Length: {length_microns:.1f} µm")

    def show_cut_image(self):
        self.crop_image()
        if self.cropped_img is not None:
            self.show_image(self.cropped_img, "Cut Image")

    def show_skeleton(self):
        self.process_image()
        if self.skeleton is not None:
            self.show_image(self.skeleton, "Skeleton", cmap="gray")

    def show_image(self, image, title, cmap=None):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=7, column=0, columnspan=2, padx=5, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
