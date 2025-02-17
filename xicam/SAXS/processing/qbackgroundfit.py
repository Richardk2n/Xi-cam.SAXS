from xicam.plugins import ProcessingPlugin, Input, Output, InOut, PlotHint
import numpy as np
from astropy.modeling import fitting, models
from astropy.modeling import Fittable1DModel

from typing import Tuple
from enum import Enum
from xicam.plugins import manager as pluginmanager


class QBackgroundFit(ProcessingPlugin):
    name = 'Q Background Fit'

    q = InOut(description='Q bin center positions',
              type=np.array)
    Iq = InOut(description='Q spectra bin intensities', type=np.array)
    # model = Input(description='Fittable model class in the style of Astropy', type=Enum)
    domainmin = Input(description='Min bound on the domain of the input data', type=float)
    domainmax = Input(description='Max bound on the domain of the input data', type=float)
    degree = Input(name='Polynomial Degree', description='Polynomial degree number', type=int, min=1, default=4)
    # fitter = Input(description='Fitting algorithm', default=fitting.LevMarLSQFitter(), type=Enum, limits={'Linear LSQ':fitting.LinearLSQFitter(), 'Levenberg-Marquardt LSQ':fitting.LevMarLSQFitter(), 'SLSQP LSQ':fitting.SLSQPLSQFitter(), 'Simplex LSQ':fitting.SimplexLSQFitter()})
    domainfilter = Input(description='Domain limits where peaks will be fitted; auto-populated by ')

    backgroundmodel = Output(description='A new model with the fitted parameters; behaves as parameterized function',
                             type=Fittable1DModel)
    backgroundprofile = Output(description='The fitted profile from the evaluation of the '
                                           'resulting model over the input range.')
    rawIq = Output(description='The spectra data before subtraction.')

    hints = [PlotHint(q, Iq), PlotHint(q, backgroundprofile), PlotHint(q, rawIq)]

    modelvars = {}

    def __init__(self):
        super(QBackgroundFit, self).__init__()
        self.peakranges = []

    @property
    def parameter(self):
        self._workflow.attach(self.find_peak_ranges)  # order may be bad...
        return super(QBackgroundFit, self).parameter

    def find_peak_ranges(self):
        from xicam.SAXS.processing.astropyfit import \
            AstropyQSpectraFit  # must be a late import to avoid being picked up first by plugin manager
        thisindex = self._workflow.processes.index(self)
        self.peakranges = [(process.domainmin.value, process.domainmax.value)
                           for process in self._workflow.processes[thisindex + 1:]
                           if isinstance(process, AstropyQSpectraFit)]

    def detach(self):
        self._workflow.detach(self.find_peak_ranges)

    def evaluate(self):
        model = models.Polynomial1D(degree=self.degree.value)

        norange = self.domainmin.value == self.domainmax.value
        if self.domainmin.value is None and self.q.value is not None or norange:  # truncate the q and I arrays with limits
            self.domainmin.value = self.q.value.min()
        if self.domainmax.value is None and self.q.value is not None or norange:  # truncate the q and I arrays with limits
            self.domainmax.value = self.q.value.max()

        filter = np.logical_and(self.domainmin.value <= self.q.value, self.q.value <= self.domainmax.value)
        for peakrange in self.peakranges:
            print('applying peak range:', peakrange)
            filter &= np.logical_or(peakrange[0] >= self.q.value, self.q.value >= peakrange[1])

        q = self.q.value[filter]
        Iq = self.Iq.value[filter]
        self.backgroundmodel.value = fitting.LinearLSQFitter()(model, q, Iq)
        self.backgroundprofile.value = self.backgroundmodel.value(self.q.value)
        self.rawIq.value = self.Iq.value.copy()
        self.Iq.value = self.Iq.value - self.backgroundprofile.value

    def getCategory() -> str:
        return "Fits"
