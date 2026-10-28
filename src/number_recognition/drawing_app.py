"""Tkinter drawing interface backed by the shared inference module."""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw
import torch
from torch import nn

from .inference import predict_image

CANVAS_SIZE = 280
BRUSH_WIDTH = 20


class DigitDrawingApp:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.root = tk.Tk()
        self.root.title("Dessiner un chiffre")
        self.root.resizable(False, False)

        self.drawing = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.painter = ImageDraw.Draw(self.drawing)
        self.previous_point: tuple[int, int] | None = None
        self.has_stroke = False

        ttk.Label(self.root, text="Dessine un chiffre de 0 à 9").pack(pady=(10, 6))
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background="white",
            cursor="crosshair",
            highlightthickness=1,
            highlightbackground="#a0a0a0",
        )
        self.canvas.pack(padx=12)
        self.canvas.bind("<ButtonPress-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._finish_stroke)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=10)
        ttk.Button(controls, text="Effacer", command=self.clear).pack(side="left")
        ttk.Button(controls, text="Prédire", command=self.predict).pack(side="left", padx=8)
        ttk.Button(controls, text="Fermer", command=self.root.destroy).pack(side="right")

        self.result = ttk.Label(self.root, text="", anchor="center")
        self.result.pack(fill="x", padx=12, pady=(0, 10))

    def _start_stroke(self, event: tk.Event) -> None:
        point = (event.x, event.y)
        radius = BRUSH_WIDTH // 2
        self.canvas.create_oval(
            point[0] - radius,
            point[1] - radius,
            point[0] + radius,
            point[1] + radius,
            fill="black",
            outline="black",
        )
        self.painter.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=0,
        )
        self.previous_point = point
        self.has_stroke = True
        self.result.configure(text="")

    def _continue_stroke(self, event: tk.Event) -> None:
        if self.previous_point is None:
            return

        point = (event.x, event.y)
        self.canvas.create_line(
            *self.previous_point,
            *point,
            fill="black",
            width=BRUSH_WIDTH,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.painter.line(
            (self.previous_point, point),
            fill=0,
            width=BRUSH_WIDTH,
        )
        radius = BRUSH_WIDTH // 2
        self.painter.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=0,
        )
        self.previous_point = point

    def _finish_stroke(self, _event: tk.Event) -> None:
        self.previous_point = None

    def clear(self) -> None:
        self.canvas.delete("all")
        self.drawing = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.painter = ImageDraw.Draw(self.drawing)
        self.has_stroke = False
        self.result.configure(text="")

    def predict(self) -> None:
        if not self.has_stroke:
            self.result.configure(text="Dessine d'abord un chiffre.")
            return

        digit, confidence = predict_image(self.model, self.drawing, self.device)
        self.result.configure(
            text=f"Chiffre prédit : {digit}  |  Confiance : {confidence:.1%}"
        )

    def run(self) -> None:
        self.root.mainloop()


def run_drawing_app(model: nn.Module, device: torch.device) -> None:
    """Open the drawing window and classify sketches with the supplied model."""
    DigitDrawingApp(model, device).run()
