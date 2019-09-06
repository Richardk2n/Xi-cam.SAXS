from xicam.plugins import ProcessingPlugin, Input, Output, PlotHint
import numpy as np
from pyFAI import AzimuthalIntegrator, units


class LinecutPlugin(ProcessingPlugin):
    name = 'Linecut'
    coordinate = Input(description='Coordinate of the cut', type = int, default = 0)
    parallelAxis = Input(description='Axis the cut is parallel to', type = str, default = "x")
    data = Input(description='2d array representing intensity for each pixel',
                 type=np.ndarray)
    mask = Input(description='Array (same size as image) with 1 for masked pixels, and 0 for valid pixels',
                 type=np.ndarray)
    dark = Input(description='Dark noise image',
                 type=np.ndarray)
    flat = Input(description='Flat field image',
                 type=np.ndarray)
    normalization_factor = Input(description='Value of a normalization monitor',
                                 type=float, default=1.)
    px = Output(name = 'px', description='Bin center positions',
                type=np.array)
    I = Output(name = 'Intensity', description='Binned/pixel-split intensity',
                type=np.array)

    hints = [PlotHint(px, I)]

    def evaluate(self):
        x = self.parallelAxis.value == "x"
        lperp = self.data.value.shape[not x]#booleans are ints
        if self.coordinate.value >= lperp:
            self.coordinate.value = lperp-1
        if self.coordinate.value < 0:
            self.coordinate.value = 0
        if self.dark.value is None: self.dark.value = np.zeros_like(self.data.value)
        if self.flat.value is None: self.flat.value = np.ones_like(self.data.value)
        if self.mask.value is None: self.mask.value = np.zeros_like(self.data.value)
        h = ((self.data.value - self.dark.value) * np.average(self.flat.value - self.dark.value) / (
                self.flat.value - self.dark.value) * np.logical_not(self.mask.value))
        self.I.value = (h[lperp -1 - self.coordinate.value] if x else [b[self.coordinate.value] for b in h][::-1])
        self.px.value = range(self.data.value.shape[x])#booleans are ints

    def getCategory() -> str:
        return "Cuts"
