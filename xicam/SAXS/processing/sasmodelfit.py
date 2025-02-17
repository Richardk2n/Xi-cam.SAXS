from xicam.plugins import ProcessingPlugin, Input, Output, InOut, PlotHint
import numpy as np
from astropy.modeling import fitting
from astropy.modeling import Fittable1DModel
from typing import Tuple
from enum import Enum
from xicam.plugins import manager as pluginmanager


class AstropyQSpectraFit(ProcessingPlugin):
    name = 'Q Fit (Astropy)'

    q = InOut(description='Q bin center positions',
              type=np.array)
    Iq = InOut(description='Q spectra bin intensities', type=np.array)
    model = Input(description='Fittable model class in the style of Astropy', type=Enum)
    domainmin = Input(description='Min bound on the domain of the input data', type=float)
    domainmax = Input(description='Max bound on the domain of the input data', type=float)
    fitter = Input(description='Fitting algorithm', default=fitting.LevMarLSQFitter(), type=Enum,
                   limits={'Linear LSQ': fitting.LinearLSQFitter(),
                           'Levenberg-Marquardt LSQ': fitting.LevMarLSQFitter(), 'SLSQP LSQ': fitting.SLSQPLSQFitter(),
                           'Simplex LSQ': fitting.SimplexLSQFitter()})

    fittedmodel = Output(description='A new model with the fitted parameters; behaves as parameterized function',
                         type=Fittable1DModel)
    fittedprofile = Output(
        description='The fitted profile from the evaluation of the resulting model over the input range.')

    hints = [PlotHint(q, Iq), PlotHint(q, fittedprofile)]

    modelvars = {}

    def __init__(self):
        super(AstropyQSpectraFit, self).__init__()
        self.model.limits = {plugin.name: plugin.plugin_object for plugin in
                             pluginmanager.getPluginsOfCategory('Fittable1DModelPlugin')}
        self.model.value = list(self.model.limits.values())[0]

    @property
    def parameter(self):

        # clear cache in
        for input in self.modelvars:
            del self.__dict__[input]
        varcache = self.modelvars.copy()
        self.modelvars = {}
        self._inputs = None
        self._inverted_vars = None
        if hasattr(self, '_inverted_vars'): del self._inverted_vars

        for name in self.model.value.param_names:
            param = getattr(self.model.value, name)
            # TODO: CHECK NAMESPACE
            if name in varcache:
                input = varcache[name]
            else:
                input = InOut(name=name, default=param.default, limits=param.bounds, type=float, fixed=False,
                              fixable=True)
            setattr(self, name, input)
            self.modelvars[name] = input
        parameter = super(AstropyQSpectraFit, self).parameter
        parameter.child('model').sigValueChanged.connect(self.reset_parameter)
        return parameter

    def reset_parameter(self):
        # cache old parameter
        oldparam = self._param

        # empty it
        for child in oldparam.children():
            child.remove()

        # reset attribute so new parameter is generated
        self._param = None

        # add new children to old parameter
        for child in self.parameter.children():  # type: Parameter
            oldparam.addChild(child)

        # set old parameter to attribute
        self._param = oldparam

    def evaluate(self):
        if self.model.value is None or self.model.value == '----': return
        norange = self.domainmin.value == self.domainmax.value
        if self.domainmin.value is None and self.q.value is not None or norange:  # truncate the q and I arrays with limits
            self.domainmin.value = self.q.value.min()
        if self.domainmax.value is None and self.q.value is not None or norange:  # truncate the q and I arrays with limits
            self.domainmax.value = self.q.value.max()
        for name, input in self.modelvars.items():  # propogate user-defined values to the model
            getattr(self.model.value, name).value = input.value
            getattr(self.model.value, name).fixed = input.fixed
        filter = np.logical_and(self.domainmin.value <= self.q.value, self.q.value <= self.domainmax.value)
        q = self.q.value[filter]
        Iq = self.Iq.value[filter]
        self.fittedmodel.value = self.fitter.value(self.model.value, q, Iq)
        self.fittedprofile.value = self.fittedmodel.value(self.q.value)

    def getCategory() -> str:
        return "Fits"
