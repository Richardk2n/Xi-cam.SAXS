from qtpy.QtGui import *
from qtpy.QtCore import Signal
from qtpy.QtWidgets import *
from xicam.gui.static import path
from pyFAI import detectors
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.multi_geometry import MultiGeometry

from xicam.plugins import ParameterSettingsPlugin
from .CalibrationPanel import CalibrationPanel

from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter, ListParameter, SimpleParameter


# https://stackoverflow.com/questions/20866996/how-to-compress-slot-calls-when-using-queued-connection-in-qt

# TODO: Refactor this class to be a view on the AI
class DeviceParameter(GroupParameter):
    def __init__(self, device, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        opts['name'] = device
        ALL_DETECTORS = {(getattr(detector, 'aliases', ) or [detector.__name__])[0]: detector for detector in
                         detectors.ALL_DETECTORS.values()}
        geometrystyle = ListParameter(name='Geometry Style', type='list',
                                      values=['Fit2D', 'pyFAI', 'wxDiff'], value='Fit2D')
        detector = ListParameter(name='Detector', type='list', values=ALL_DETECTORS, value=ALL_DETECTORS['Pilatus 2M'])
        pixelx = SimpleParameter(name='Pixel Size X', type='float', value=172.e-6, siPrefix=True, suffix='m')
        pixely = SimpleParameter(name='Pixel Size Y', type='float', value=172.e-6, siPrefix=True, suffix='m')
        binning = SimpleParameter(name='Binning', type='int', value=1, suffix='x', limits=(1, 100))
        centerx = SimpleParameter(name='Center X', type='float', value=0, suffix=' px', decimals=4)
        centery = SimpleParameter(name='Center Y', type='float', value=0, suffix=' px', decimals=4)
        sdd = SimpleParameter(name='Detector Distance', type='float', value=1, siPrefix=True, suffix='m',
                              limits=(0, 1000), step=1e-3)
        tilt = SimpleParameter(name='Detector Tilt', type='float', value=0, siPrefix=False, suffix=u'°')
        rotation = SimpleParameter(name='Detector Rotation', type='float', value=0, siPrefix=False, suffix=u'°')

        self.children = [geometrystyle, detector, pixelx, pixely, binning, centerx, centery, sdd, tilt, rotation]
        opts['children'] = self.children
        super(DeviceParameter, self).__init__(**opts)
        #     wavelengthparam = self.param('Wavelength')
        #     energyparam = self.param('Energy')
        #     wavelengthparam.sigValueChanged.connect(self.wavelengthChanged)
        #     energyparam.sigValueChanged.connect(self.energyChanged)
        #
        #


class DeviceProfiles(ParameterSettingsPlugin):
    sigGeometryChanged = Signal(AzimuthalIntegrator)  # Emits the new geometry
    sigSimulateCalibrant = Signal()

    def __init__(self):
        self.headermodel = None
        self.selectionmodel = None
        self.multiAI = MultiGeometry([])
        self.AIs = dict()


        energy = SimpleParameter(name='Energy', type='float', value=10000, siPrefix=True, suffix='eV')
        wavelength = SimpleParameter(name='Wavelength', type='float', value=1.239842e-6 / 10000, siPrefix=True,
                                     suffix='m')
        incidentAngle = SimpleParameter(name='Incident Angle', type='float', value=90, siPrefix=False, suffix=u'°', limits = (0, 90))
        reflection = SimpleParameter(name='Reflection', type='int', value=0, siPrefix=False, limits = (0, 1), step = 1)

        icon = QIcon(str(path('icons/calibrate.png')))
        super(DeviceProfiles, self).__init__(icon, "Device Profiles", [energy, wavelength, incidentAngle, reflection], addText='New Device')

        self.sigTreeStateChanged.connect(self.simulateCalibrant)
        self.sigTreeStateChanged.connect(self.genAIs)
        self.sigTreeStateChanged.connect(self.geometryChanged)
        self.sigGeometryChanged.connect(self.save)

    def addNew(self, typ=None):
        text, ok = QInputDialog().getText(self.widget, 'Enter Device Name', 'Device Name:')

        if text and ok:
            self.addDevice(text)


    def geometryChanged(self, A, B):
        if B[0][0].parent():
            name = B[0][0].parent().name()
            self.sigGeometryChanged.emit(self.AI(name))

    def simulateCalibrant(self, *args):
        self.sigSimulateCalibrant.emit()

    def genAIs(self, parent, changes):
        for parameter, key, value in changes:
            if parameter.name == 'Wavelength':
                self.param('Energy').setValue(1.239842e-6 / self.param('Wavelength').value(),
                                              blockSignal=self.parameter.sigTreeStateChanged)
            elif parameter.name == 'Energy':
                self.param('Wavelength').setValue(1.239842e-6 / self.param('Energy').value(),
                                                  blockSignal=self.WavelengthChanged)

        for parameter in self.children():
            if isinstance(parameter, DeviceParameter):
                device = parameter.name()
                ai = self.AI(device)
                ai.set_wavelength(self['Wavelength'])
                ai.detector = parameter['Detector']()
                ai.detector.set_binning([parameter['Binning']] * 2)
                ai.detector.set_pixel1(parameter['Pixel Size X'])
                ai.detector.set_pixel2(parameter['Pixel Size Y'])
                fit2d = ai.getFit2D()
                fit2d['centerX'] = parameter['Center X']
                fit2d['centerY'] = parameter['Center Y']
                fit2d['directDist'] = parameter['Detector Distance'] * 1000
                fit2d['tilt'] = parameter['Detector Tilt']
                fit2d['tiltPlanRotation'] = parameter['Detector Rotation']
                ai.setFit2D(**fit2d)

    def AI(self, device):
        if device not in self.AIs:
            self.addDevice(device)
        return self.AIs.get(device, None)

    def setAI(self, ai: AzimuthalIntegrator, device: str):
        self.AIs[device] = ai
        self.multiAI.ais = self.AIs.values()

        # propagate new ai to parameter
        fit2d = ai.getFit2D()
        try:
            self.setSilence(True)
            self.child(device, 'Detector').setValue(type(ai.detector))
            self.child(device, 'Binning').setValue(ai.detector.binning[0])
            self.child(device, 'Detector Tilt').setValue(fit2d['tiltPlanRotation'])
            self.child(device, 'Detector Rotation').setValue(fit2d['tilt'])
            self.child(device, 'Pixel Size X').setValue(ai.pixel1)
            self.child(device, 'Pixel Size Y').setValue(ai.pixel2)
            self.child(device, 'Center X').setValue(fit2d['centerX'])
            self.child(device, 'Center Y').setValue(fit2d['centerY'])
            self.child(device, 'Detector Distance').setValue(fit2d['directDist'] / 1000.)
            self.child('Wavelength').setValue(ai.wavelength)
        finally:
            self.setSilence(False)
            self.simulateCalibrant()
            self.sigGeometryChanged.emit(ai)

    def setSilence(self, silence):
        if silence:
            try:
                self.sigTreeStateChanged.disconnect(self.simulateCalibrant)
                self.sigTreeStateChanged.disconnect(self.genAIs)
                self.sigTreeStateChanged.disconnect(self.geometryChanged)
            except TypeError:
                pass  # do nothing if no connected
        else:
            self.sigTreeStateChanged.connect(self.simulateCalibrant)
            self.sigTreeStateChanged.connect(self.genAIs)
            self.sigTreeStateChanged.connect(self.geometryChanged)


    def addDevice(self, device):
        if device:
            try:
                self.setSilence(True)
                devicechild = DeviceParameter(device)
                self.addChild(devicechild)
                ai = AzimuthalIntegrator(wavelength=self['Wavelength'])
                ai.detector = detectors.Pilatus2M()
                self.AIs[device] = ai
                self.multiAI.ais = list(self.AIs.values())
            finally:
                self.setSilence(False)

    def setModels(self, headermodel, selectionmodel):
        self.headermodel = headermodel
        self.headermodel.dataChanged.connect(self.dataChanged)
        self.selectionmodel = selectionmodel


    def dataChanged(self, start, end):
        devices = self.headermodel.item(self.selectionmodel.currentIndex().row()).header.devices()
        for device in devices:
            if device not in self.AIs:
                self.addDevice(device)

    def wavelengthChanged(self):
        self.param('Energy').setValue(1.239842e-6 / self.param('Wavelength').value(), blockSignal=self.EnergyChanged)

    def energyChanged(self):
        self.param('Wavelength').setValue(1.239842e-6 / self.param('Energy').value(),
                                          blockSignal=self.WavelengthChanged)

    def toState(self):
        self.apply()
        return self.saveState(filter='user'), self.AIs

    def fromState(self, state):
        self.restoreState(state[0], addChildren=False, removeChildren=False)
        self.AIs = state[1]
        for child in self.children()[4:]: #4 == Amount of not device children. Here energy wavelength, incident angle and reflection
            child.remove()
        for name, ai in self.AIs.items():
            self.addDevice(name)
            self.setAI(ai, name)
