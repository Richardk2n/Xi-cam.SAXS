from qtpy.QtWidgets import *
from qtpy.QtCore import *
from qtpy.QtGui import *
from xicam.plugins.widgetplugin import QWidgetPlugin
from xicam.gui.static import path
from xicam.core.execution.workflow import Workflow
from xicam.plugins import ProcessingPlugin, Output
from xicam.gui.widgets.menuview import MenuView
from xicam.plugins import Hint
from functools import partial


class SAXSToolbar(QToolBar, QWidgetPlugin):
    name = 'SAXSToolbar'
    sigPlotCache = Signal()
    sigDoWorkflow = Signal()
    sigDeviceChanged = Signal(str)

    def __init__(self, headermodel: QStandardItemModel, selectionmodel: QItemSelectionModel):
        super(SAXSToolbar, self).__init__()

        self.results = []

        self.headermodel = headermodel
        self.selectionmodel = selectionmodel
        self.headermodel.dataChanged.connect(self.updatedetectorcombobox)

        self.detectorcombobox = QComboBox()
        self.detectorcombobox.currentTextChanged.connect(self.sigDeviceChanged)

        self.addWidget(self.detectorcombobox)
        self.addSeparator()
        self.modegroup = QActionGroup(self)
        self.rawaction = self.mkAction('icons/raw.png', 'Raw', checkable=True, group=self.modegroup, checked=True)
        self.addAction(self.rawaction)
        self.cakeaction = self.mkAction('icons/cake.png', 'Cake (q/chi plot)', checkable=True, group=self.modegroup)
        self.addAction(self.cakeaction)
        self.remeshaction = self.mkAction('icons/remesh.png', 'Remesh (GIWAXS)', checkable=True, group=self.modegroup)
        self.addAction(self.remeshaction)
        self.addSeparator()

        self.multiplot = QAction(self)
        self.multiplot.setIcon(QIcon(str(path('icons/multiplot.png'))))
        self.multiplot.setText('Plot Series')
        self.multiplot.setCheckable(True)
        self.multiplot.triggered.connect(self.sigDoWorkflow)
        self.addAction(self.multiplot)

    # def updateReductionModes(self, results):
    #     previousindex = self.reductionModes.currentIndex()
    #     self.reductionModes.currentIndexChanged.disconnect(self.sigPlotCache)
    #     self.reductionModesModel.clear()
    #     for result in results:
    #         for key, output in result.items():
    #             for xname in output.hints.get('plotx', []):
    #                 name = f'{result[key].name} vs. {xname} [{result[key].parent.name}]'
    #                 item = QStandardItem(name)
    #                 item.setData((result[xname], result[key]), role=256)
    #                 item.setCheckState(Qt.Unchecked)
    #                 self.reductionModesModel.appendRow(item)
    #     self.reductionModes.setCurrentIndex(previousindex)
    #     self.reductionModes.currentIndexChanged.connect(self.sigPlotCache)
    #     if previousindex == -1:
    #         self.reductionModes.setCurrentIndex(0)
    #         self.reductionModesModel.item(0).setCheckState(Qt.Checked)


    def updatedetectorcombobox(self, start, end):
        if self.headermodel.rowCount():
            devices = self.headermodel.item(self.selectionmodel.currentIndex().row()).header.devices()
            self.detectorcombobox.clear()
            self.detectorcombobox.addItems(devices)

    def mkAction(self, iconpath: str = None, text=None, receiver=None, group=None, checkable=False, checked=False):
        actn = QAction(self)
        if iconpath: actn.setIcon(QIcon(QPixmap(str(path(iconpath)))))
        if text: actn.setText(text)
        if receiver: actn.triggered.connect(receiver)
        actn.setCheckable(checkable)
        if checked: actn.setChecked(checked)
        if group: actn.setActionGroup(group)
        return actn


class CheckableWorkflowOutputModel(QAbstractItemModel):
    def __init__(self, workflow: Workflow, *args):
        super(CheckableWorkflowOutputModel, self).__init__(*args)
        self.workflow = workflow
        self.workflow.attach(partial(self.modelReset.emit))

    def index(self, row, column, parent=None):
        if parent is None or not parent.isValid():
            if row > len(self.workflow.processes) - 1: return QModelIndex()
            return self.createIndex(row, column, self.workflow.processes[row])

        parentNode = parent.internalPointer()

        if isinstance(parentNode, ProcessingPlugin):
            return self.createIndex(row, column, parentNode.hints[row])
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        if isinstance(node, ProcessingPlugin):
            return QModelIndex()
        if isinstance(node, Hint):
            if node.parent not in self.workflow.processes: return QModelIndex()
            return self.createIndex(self.workflow.processes.index(node.parent), 0, node.parent)

        return QModelIndex()

    def rowCount(self, parent=None, *args, **kwargs):

        if parent is None or not parent.isValid():
            return len(self.workflow.processes)

        node = parent.internalPointer()
        if isinstance(node, ProcessingPlugin):
            return len(node.hints)

        return 0

    def columnCount(self, parent=None, *args, **kwargs):
        return 1

    def data(self, index: QModelIndex, role):
        if role == Qt.DisplayRole:
            return index.internalPointer().name
        elif role == Qt.CheckStateRole and isinstance(index.internalPointer(), Hint):
            return index.internalPointer().checked

    def setData(self, index: QModelIndex, value, role=Qt.EditRole):
        if role == Qt.CheckStateRole:
            index.internalPointer().checked = value

    def flags(self, index):
        if index.parent().isValid():  # if index is a hint
            return Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        elif index.internalPointer().hints:  # if index is a process with hints
            return Qt.ItemIsEnabled
        else:
            return

    def checkedIndices(self):
        return [self.index(hintindex, 0, self.index(procindex, 0))
                for procindex in range(self.rowCount())
                for hintindex in range(self.rowCount(self.index(procindex, 0)))
                if self.data(self.index(hintindex, 0, self.index(procindex, 0)), Qt.CheckStateRole)]
