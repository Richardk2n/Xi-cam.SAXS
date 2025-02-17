import numpy as np
from xicam.plugins import ProcessingPlugin, Input, Output, InOut
from pyFAI import AzimuthalIntegrator, calibrant


class SimulateCalibrant(ProcessingPlugin):
    name = 'Simulate Calibrant'
    ai = Input(description='Azimimuthal integrator',
               type=AzimuthalIntegrator)
    Imax = Input(description='Maximum intensity of rings', type=float, default=1)
    calibrant = Input(description='pyFAI calibrant object to simulate', type=calibrant)
    data = Output(description='Simulated data for given calibrant', type=np.ndarray)

    def evaluate(self):
        self.calibrant.value.set_wavelength(self.ai.value.get_wavelength())
        self.data.value = self.calibrant.value.fake_calibration_image(self.ai.value, Imax=self.Imax.value)
        d = []
        for i in range(len(self.data.value[0])):
            c = []
            for j in range(len(self.data.value)-1, -1, -1):
                c.append(self.data.value[j][i])
            d.append(np.asarray(c))
        d = np.asarray(d)
        self.data.value = d
