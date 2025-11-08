# predict user-drawn digits using trained ANN model on a Tkinter canvas
import tkinter as tk
import pickle
import numpy as np
from scipy.ndimage import shift
from train import NeuralNetwork, load_data, one_hot_encode, softmax, relu, relu_derivative

def center_digit(image):
    """
    Center a digit within a 28*28 grid.
    """

    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    if not rows.any() or not cols.any(): # empty 
        return image
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # calculate center
    row_center = (row_min + row_max) // 2
    col_center = (col_min + col_max) // 2

    # calculate shift amount
    row_shift = 14 - row_center
    col_shift = 14 - col_center

    # shift image
    centered_image = shift(image, shift=(row_shift, col_shift), mode='constant', cval=0)
    return centered_image

def process(image):
    """
    Process the input image for digit recognition.
    """
    # Center the digit
    centered = center_digit(image)

    # Flatten the image
    flattened = centered.flatten()

    # binarize the image
    binarized = (flattened > 0.5).astype(np.float32)

    return binarized

class PredictCanvas:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Predictor")

        # Constants
        self.GRID_SIZE = 28
        self.CELL_SIZE = 10

        # Instructions label
        self.label = tk.Label(master, text="Draw, Enter to predict, Delete to clear.", font=("Arial", 14))
        self.label.pack(pady=5)
        
        # Main frame for two panels
        main_frame = tk.Frame(master)
        main_frame.pack(padx=10, pady=10)
        
        # Drawing canvas (left panel)
        self.canvas = tk.Canvas(main_frame, width=self.GRID_SIZE*self.CELL_SIZE, height=self.GRID_SIZE*self.CELL_SIZE, bg='white', bd=2, relief='solid')
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))
        
        # Prediction display canvas (right panel)
        self.prediction_canvas = tk.Canvas(main_frame, width=self.GRID_SIZE*self.CELL_SIZE, height=self.GRID_SIZE*self.CELL_SIZE, bg='lightgray', bd=2, relief='solid')
        self.prediction_canvas.pack(side=tk.LEFT)
        
        # Initial prediction canvas text
        self.prediction_canvas.create_text(self.GRID_SIZE*self.CELL_SIZE//2, self.GRID_SIZE*self.CELL_SIZE//2,
                                         text="Predicted digit\nwill appear here",
                                         font=("Arial", 16), justify=tk.CENTER)
        
        # Confidence display
        self.confidence_label = tk.Label(master, text="", font=("Arial", 12))
        self.confidence_label.pack(pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictCanvas(root)
    root.mainloop()