# predict user-drawn digits using trained ANN model on a Tkinter canvas
import tkinter as tk
import pickle
import numpy as np
from scipy.ndimage import shift
from train import NeuralNetwork, load_data, one_hot_encode, softmax, relu, relu_derivative

def center_digit(matrix):
    """
    Center a digit within a 28*28 grid.
    """

    rows = np.any(matrix, axis=1)
    cols = np.any(matrix, axis=0)
    if not rows.any() or not cols.any(): # empty 
        return matrix
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # calculate center
    row_center = (row_min + row_max) // 2
    col_center = (col_min + col_max) // 2

    # calculate shift amount
    row_shift = 14 - row_center
    col_shift = 14 - col_center

    # shift matrix
    centered_matrix = shift(matrix, shift=(row_shift, col_shift), mode='constant', cval=0)
    return centered_matrix

def process(matrix):
    """
    Process the input matrix for digit recognition.
    """
    # Center the digit
    centered = center_digit(matrix)

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

        # Initialize drawing matrix
        self.matrix = np.zeros((self.GRID_SIZE, self.GRID_SIZE))

        # Binds
        self.canvas.bind("<B1-Motion>", self.paint)  # Mouse drag
        self.canvas.bind("<Button-1>", self.paint)   # Mouse click
        self.master.bind("<Return>", self.predict_digit)  # Predict
        self.master.bind("<BackSpace>", self.clear_canvas)  # Clear

        # Autoclear flag
        self.should_clear = False

        # Load trained model
        with open("model_final.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.sizes = self.model['sizes']
        self.weights = [np.array(w) for w in self.model['weights']]
        self.biases = [np.array(b).reshape(-1, 1) for b in self.model["biases"]]


    def paint(self, event):
        """
        Paint a 2x2 block on the canvas and update the matrix
        """
        # Clear board after prediction
        if self.should_clear:
            self.clear_canvas()
            self.should_clear = False

        x, y = event.x // self.CELL_SIZE, event.y // self.CELL_SIZE

        for dy in range(2):
            for dx in range(2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                    self.matrix[ny][nx] = 1.0
                    self.canvas.create_rectangle(
                        nx*self.CELL_SIZE, ny*self.CELL_SIZE,
                        (nx+1)*self.CELL_SIZE, (ny+1)*self.CELL_SIZE,
                        fill='black', outline='black'
                    )

    def clear_canvas(self):
        """
        Clear the canvas and reset the matrix
        """
        self.canvas.delete("all")
        self.prediction_canvas.delete("all")
        self.prediction_canvas.create_text(self.GRID_SIZE*self.CELL_SIZE//2, self.GRID_SIZE*self.CELL_SIZE//2,
                                         text="Predicted digit\nwill appear here",
                                         font=("Arial", 16), justify=tk.CENTER)
        self.confidence_label.config(text="")
        self.matrix = np.zeros((self.GRID_SIZE, self.GRID_SIZE))

    def predict_digit(self, event=None):
        """
        Predict the digit drawn on the canvas
        """
        # preprocess the matrix
        preprocessed = process(self.matrix)
        features = preprocessed.reshape(-1, 1)

        # forward pass
        activation = features
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activation) + b
            activation = relu(z)
        
        # output layer with scaling
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        z_scaled = z * 2.0
        probs = softmax(z_scaled.flatten())

        digit = np.argmax(probs)
        confidence = probs[digit]

        # Dispay
        self.prediction_canvas.delete("all")
        self.prediction_canvas.create_text(self.GRID_SIZE*self.CELL_SIZE//2, self.GRID_SIZE*self.CELL_SIZE//2, 
                                         text=str(digit), 
                                         font=("Arial", 120, "bold"), fill="blue")
        self.confidence_label.config(text=f"Confidence: {confidence*100:.2f}%")
        self.should_clear = True




if __name__ == "__main__":
    root = tk.Tk()
    app = PredictCanvas(root)
    root.mainloop()