import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode, vtkMRMLMarkupsFiducialNode

import numpy as np

#
# ContactDetector
#


class ContactDetector(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Contact Detector")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "SEEG")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Jakub Smid (CTU in Prague)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#ContactDetector">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # ContactDetector1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="ContactDetector",
        sampleName="ContactDetector1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "ContactDetector1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="ContactDetector1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="ContactDetector1",
    )

    # ContactDetector2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="ContactDetector",
        sampleName="ContactDetector2",
        thumbnailFileName=os.path.join(iconsPath, "ContactDetector2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="ContactDetector2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="ContactDetector2",
    )


#
# ContactDetectorParameterNode
#


@parameterNodeWrapper
class ContactDetectorParameterNode:
    """
    The parameters needed by module.
    """

    inputCT: vtkMRMLScalarVolumeNode
    inputT1: vtkMRMLScalarVolumeNode
    brainMask: vtkMRMLSegmentationNode
    boltFiducials: vtkMRMLMarkupsFiducialNode

    metalThreshold_HU: Annotated[float, WithinRange(0, 9999999)] = 3000
    contactLength_mm: Annotated[float, WithinRange(0.1, 100)] = 2
    contactGap_mm: Annotated[float, WithinRange(0.1, 100)] = 1.5
    contactDiameter_mm: Annotated[float, WithinRange(0.1, 100)] = 0.8
    boltSphereRadius_mm: Annotated[float, WithinRange(0.1, 100)] = 10
    blobSize_sigma: Annotated[float, WithinRange(0.1, 100)] = 3

    # developerMode
    boltSegmentation: bool
    linearApproximation: bool
    electrodeSegmentation: bool
    gaussianBalls: bool

class Electrode():
    def __init__(self, bolt_tip_ras, label, contact_length_mm, contact_gap_mm):
        self.bolt_tip_ras = np.array(bolt_tip_ras)
        self.label = label
        self.label_prefix, self.n_contacts = Electrode.split_label(label)
        self.length_mm = contact_length_mm * self.n_contacts + contact_gap_mm * (self.n_contacts - 1)
        
        self.bolt_segmentation_indices_ijk = None

        self.tip_ijk = None
        self.entry_point_ijk = None
        self.gmm_segmentation_indices_ijk = None

        self.curve_points = None
        self.curve_cumulative_distances = None
        self.curve_points_offset = None
        self.shift_fiducials_value = 0
    
    @staticmethod
    def split_label(electrode_name):
        split = electrode_name.split("-")
        return split[0], int(split[1])

#
# ContactDetectorWidget
#


class ContactDetectorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        # VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.lastSelectedBoltFiducials = None

        self.electrodes: list[Electrode] = []
        self.selected_electrode = None
        self.selected_markup_index = None
        self.result_markup_node = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ContactDetector.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        if self.developerMode == False:
            self.ui.collapsibleButtonDebug.setVisible(False)

        self.ui.shiftFiducialsWidget.setVisible(False)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ContactDetectorLogic()

        # Connections

        self.ui.comboBoxCT.connect('currentNodeChanged(vtkMRMLNode*)', self.onComboBoxCTChanged)
        self.ui.SimpleMarkupsWidgetEstimatedContacts.connect('currentMarkupsControlPointSelectionChanged(int)', self.onMarkupsWidgetEstimatedContactsSelectionChanged)
        self.ui.SimpleMarkupsWidgetEstimatedContacts.connect('markupsNodeChanged()', self.onMarkupsWidgetEstimatedContactsMarkupsNodeChanged)
        self.ui.spinBoxShiftElectrode.connect('valueChanged(int)', self.onSpinBoxShiftElectrodeChanged)

        # Buttons
        self.ui.buttonDisplayCT.connect("clicked(bool)", self.onDisplayCTClicked)

        self.ui.radioButtonRenderingMetal.connect("clicked(bool)", self.onRenderingMetalClicked)
        self.ui.radioButtonRenderingHead.connect("clicked(bool)", self.onRenderingHeadClicked)
        self.ui.radioButtonRenderingDisabled.connect("clicked(bool)", self.onRenderingDisabledClicked)

        # developerMode
        self.ui.buttonBoltSegmentation.connect("clicked(bool)", self.onBoltSegmentationClicked)
        self.ui.buttonBoltAxisEst.connect("clicked(bool)", self.onBoltAxisEstClicked)
        self.ui.buttonElectrodeSegmentation.connect("clicked(bool)", self.onElectrodeSegmentationClicked)
        self.ui.buttonCurveFitting.connect("clicked(bool)", self.onCurveFittingClicked)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def onSpinBoxShiftElectrodeChanged(self, spin_box_value):
        fiducial_node = self.ui.SimpleMarkupsWidgetEstimatedContacts.currentNode()
        self.selected_electrode.shift_fiducials_value = spin_box_value

        # get fiducials list
        fiducial_idx = []
        for i in range(fiducial_node.GetNumberOfControlPoints()):
            if fiducial_node.GetNthControlPointLabel(i).startswith(self.selected_electrode.label_prefix):
                fiducial_idx.append(i)

        selected_points = self.logic.select_contact_points(self.selected_electrode.curve_points,
                                                           self.selected_electrode.curve_cumulative_distances,
                                                           self.selected_electrode.curve_points_offset,
                                                           self.selected_electrode.n_contacts,
                                                           self._parameterNode.contactLength_mm,
                                                           self._parameterNode.contactGap_mm,
                                                           spin_box_value)

        # pause rendering while changing fiducial positions
        with slicer.util.RenderBlocker():
            # convert to RAS
            for i, point in enumerate(selected_points):
                    selected_points[i] = self.logic.IJK_to_RAS(point[::-1], self._parameterNode.inputCT)
                    fiducial_node.SetNthControlPointPosition(fiducial_idx[i], selected_points[i])

            # jump slices to updated fiducials
            # slicer.modules.markups.logic().JumpSlicesToNthPointInMarkup(fiducial_node.GetID(), self.selected_markup_index, True)


    def onMarkupsWidgetEstimatedContactsSelectionChanged(self, markupIndex):
        fiducial_node = self.ui.SimpleMarkupsWidgetEstimatedContacts.currentNode()
        selected_label = fiducial_node.GetNthControlPointLabel(markupIndex)
        self.selected_markup_index = markupIndex

        # find electrode by name
        for electrode in self.electrodes:
            if selected_label.startswith(electrode.label_prefix):
                self.selected_electrode = electrode
                self.ui.labelShiftElectrode.setText(f"Shift {electrode.label_prefix} fiducials:")

                self.ui.spinBoxShiftElectrode.disconnect('valueChanged(int)')
                self.ui.spinBoxShiftElectrode.setValue(electrode.shift_fiducials_value)
                self.ui.spinBoxShiftElectrode.connect('valueChanged(int)', self.onSpinBoxShiftElectrodeChanged)
                break

    def onMarkupsWidgetEstimatedContactsMarkupsNodeChanged(self):
        # check for invalid markup node in the list (result markup node has been probably deleted)
        if self.ui.SimpleMarkupsWidgetEstimatedContacts.currentNode() is not self.result_markup_node:
            self.ui.shiftFiducialsWidget.setVisible(False)
            self.ui.SimpleMarkupsWidgetEstimatedContacts.setCurrentNode(None)
            self.result_markup_node = None

    def onCurveFittingClicked(self):
        with slicer.util.WaitCursor():
            self.result_markup_node = self.logic.curve_fitting(self._parameterNode.inputCT,
                                                             self.electrodes,
                                                             self._parameterNode.boltSphereRadius_mm,
                                                             self._parameterNode.blobSize_sigma,
                                                             self._parameterNode.contactDiameter_mm,
                                                             self._parameterNode.contactLength_mm,
                                                             self._parameterNode.contactGap_mm,
                                                             self._parameterNode.gaussianBalls)
        
        # collapse collapsibleButtonInputBoltFiducials and make visible output collapsible button
        self.ui.collapsibleButtonInputBoltFiducials.collapsed = True
        self.ui.collapsibleButtonEstimatedContacts.collapsed = False
        self.ui.shiftFiducialsWidget.setVisible(True)

        # show markupsNode in the list widget
        self.ui.SimpleMarkupsWidgetEstimatedContacts.setCurrentNode(self.result_markup_node)
        
        # highlight the first control point
        self.ui.SimpleMarkupsWidgetEstimatedContacts.setJumpToSliceEnabled(False)
        self.ui.SimpleMarkupsWidgetEstimatedContacts.highlightNthControlPoint(0)
        self.ui.SimpleMarkupsWidgetEstimatedContacts.setJumpToSliceEnabled(True)

    def onElectrodeSegmentationClicked(self):
        with slicer.util.WaitCursor():
            self.logic.eletrode_segmentation(self._parameterNode.inputCT,
                                             self._parameterNode.brainMask,
                                             self.electrodes,
                                             self._parameterNode.metalThreshold_HU,
                                             self._parameterNode.electrodeSegmentation)

    def onBoltAxisEstClicked(self):
        with slicer.util.WaitCursor():
            self.logic.bolt_axis_estimation(self._parameterNode.inputCT,
                                            self.electrodes,
                                            self._parameterNode.linearApproximation)

    def onBoltSegmentationClicked(self):
        # load electrodes
        self.electrodes = []
        for i in range(self._parameterNode.boltFiducials.GetNumberOfControlPoints()):
            bolt_tip_ras = [0, 0, 0]
            self._parameterNode.boltFiducials.GetNthControlPointPosition(i, bolt_tip_ras)
            self.electrodes.append(Electrode(bolt_tip_ras, self._parameterNode.boltFiducials.GetNthControlPointLabel(i), self._parameterNode.contactLength_mm, self._parameterNode.contactGap_mm))

        # segment bolts
        with slicer.util.WaitCursor():
            self.logic.bolt_segmentation(self._parameterNode.inputCT,
                                         self.electrodes,
                                         self._parameterNode.boltSphereRadius_mm,
                                         self._parameterNode.metalThreshold_HU,
                                         self._parameterNode.boltSegmentation)

    def onRenderingMetalClicked(self):
        displayNode = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(self._parameterNode.inputCT)
        if displayNode is None:
            displayNode = slicer.modules.volumerendering.logic().CreateDefaultVolumeRenderingNodes(self._parameterNode.inputCT)
        range = self._parameterNode.inputCT.GetImageData().GetScalarRange()

        scalarOpacity = displayNode.GetVolumePropertyNode().GetScalarOpacity()
        scalarOpacity.RemoveAllPoints()
        scalarOpacity.AddPoint(self._parameterNode.metalThreshold_HU, 0.0)
        scalarOpacity.AddPoint(self._parameterNode.metalThreshold_HU + 2, 1.0)
        """
        +2 to suppress warning:
        Warning: In vtkOpenGLVolumeLookupTable.cxx, line 88
        vtkOpenGLVolumeOpacityTable (0x7484f61ffc80): This OpenGL implementation does not support the required texture size of 65536, falling back to maximum allowed, 32768.This may cause an incorrect lookup table mapping.
        """
        scalarOpacity.AddPoint(range[1], 1.0)
        displayNode.SetVisibility(True)

    def onRenderingHeadClicked(self):
        displayNode = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(self._parameterNode.inputCT)
        if displayNode is None:
            displayNode = slicer.modules.volumerendering.logic().CreateDefaultVolumeRenderingNodes(self._parameterNode.inputCT)
        range = self._parameterNode.inputCT.GetImageData().GetScalarRange()

        scalarOpacity = displayNode.GetVolumePropertyNode().GetScalarOpacity()
        scalarOpacity.RemoveAllPoints()
        scalarOpacity.AddPoint(0, 0.0)
        scalarOpacity.AddPoint(range[1], 1.0)
        displayNode.SetVisibility(True)

    def onRenderingDisabledClicked(self):
        displayNode = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(self._parameterNode.inputCT)
        if displayNode is not None:
            displayNode.SetVisibility(False)

    def onComboBoxCTChanged(self):
        # disable rendering for the new CT
        if self._parameterNode.inputCT is not None:
            self.onRenderingDisabledClicked()
            self.ui.radioButtonRenderingDisabled.setChecked(True)

    def onDisplayCTClicked(self):
        slicer.util.setSliceViewerLayers(background = self._parameterNode.inputCT)
        slicer.util.resetSliceViews()

    def updateGUIFromParameterNode(self, caller, event):
        # disable skull stripping button if T1 is not selected
        if self._parameterNode.inputT1 is None:
            self.ui.buttonCreateBrainMask.setEnabled(False)
            self.ui.buttonCreateBrainMask.setToolTip("Missing T1 image")
        else:
            self.ui.buttonCreateBrainMask.setEnabled(True)
            self.ui.buttonCreateBrainMask.setToolTip("")

        # disable rendering buttons and slice views if CT is not selected
        if self._parameterNode.inputCT is None:
            self.ui.collapsibleButtonRendering.setEnabled(False)
            self.ui.collapsibleButtonRendering.setToolTip("Missing CT image")

            self.ui.buttonDisplayCT.setEnabled(False)
            self.ui.buttonDisplayCT.setToolTip("Missing CT image")
        else:
            self.ui.collapsibleButtonRendering.setEnabled(True)
            self.ui.collapsibleButtonRendering.setToolTip("")

            self.ui.buttonDisplayCT.setEnabled(True)
            self.ui.buttonDisplayCT.setToolTip("")

        # disable run button if any input is missing
        missing = []
        if self._parameterNode.inputCT is None:
            missing.append("CT")
        if self._parameterNode.brainMask is None:
            missing.append("brain mask")
        if self._parameterNode.boltFiducials is None:
            missing.append("bolt fiducials")

        if missing:
            self.ui.buttonRun.setEnabled(False)
            self.ui.buttonRun.setToolTip(f"Missing input(s): {', '.join(missing)}")
        else:
            self.ui.buttonRun.setEnabled(True)
            self.ui.buttonRun.setToolTip("")

        # update markup widgets based on selected point list
        self.ui.MarkupsPlaceWidget.setCurrentNode(self._parameterNode.boltFiducials)
        self.ui.SimpleMarkupsWidgetInput.setCurrentNode(self._parameterNode.boltFiducials)

        # add observer for changing control point label
        if self.lastSelectedBoltFiducials is not None:
            self.removeObserver(self.lastSelectedBoltFiducials, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onControlPointAdded)

        if self._parameterNode.boltFiducials is not None:
            self.lastSelectedBoltFiducials = self._parameterNode.boltFiducials
            self.addObserver(self.lastSelectedBoltFiducials, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onControlPointAdded)
            
    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeAdded(self, caller, event,  node):
        # Eevery time a new node is added to the scene check if it's input node
        if "ct" in node.GetName().lower() and isinstance(node, ContactDetectorParameterNode.__annotations__['inputCT']):
            self._parameterNode.inputCT = node

        if "t1" in node.GetName().lower() and isinstance(node, ContactDetectorParameterNode.__annotations__['inputT1']):
            self._parameterNode.inputT1 = node
    
        if "fiducials" in node.GetName().lower() and isinstance(node, ContactDetectorParameterNode.__annotations__['boltFiducials']):
            self._parameterNode.boltFiducials = node

        if "mask" in node.GetName().lower() and isinstance(node, ContactDetectorParameterNode.__annotations__['brainMask']):
            self._parameterNode.brainMask = node

    def onControlPointAdded(self, caller, event):
        # rename new control point as A-no, B-no, etc
        if self.ui.MarkupsPlaceWidget.placeModeEnabled:
            n = caller.GetNumberOfControlPoints()
            label = f"{chr(ord('A') + n - 1)}-no"
            caller.SetNthControlPointLabel(n-1, label)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
            self.removeObserver(slicer.mrmlScene, slicer.vtkMRMLScene.NodeAddedEvent, self.onNodeAdded)
            if self.hasObserver(self.lastSelectedBoltFiducials, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onControlPointAdded):
                self.removeObserver(self.lastSelectedBoltFiducials, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onControlPointAdded)
            self.lastSelectedBoltFiducials = None

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputCT:
            volume_nodes = slicer.util.getNodesByClass(ContactDetectorParameterNode.__annotations__['inputCT'].__name__)
            for node in volume_nodes:
                if "ct" in node.GetName().lower():
                    self._parameterNode.inputCT = node
                if "t1" in node.GetName().lower():
                    self._parameterNode.inputT1 = node
            
            fiducial_nodes = slicer.util.getNodesByClass(ContactDetectorParameterNode.__annotations__['boltFiducials'].__name__)
            for node in fiducial_nodes:
                if "fiducials" in node.GetName().lower():
                    self._parameterNode.boltFiducials = node

            mask_nodes = slicer.util.getNodesByClass(ContactDetectorParameterNode.__annotations__['brainMask'].__name__)
            for node in mask_nodes:
                if "mask" in node.GetName().lower():
                    self._parameterNode.brainMask = node

    def setParameterNode(self, inputParameterNode: Optional[ContactDetectorParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)

            # remove observers from the old parameter node
            if self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
                self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
            if self.hasObserver(slicer.mrmlScene, slicer.vtkMRMLScene.NodeAddedEvent, self.onNodeAdded):
                self.removeObserver(slicer.mrmlScene, slicer.vtkMRMLScene.NodeAddedEvent, self.onNodeAdded)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

            # add observer for GUI changes
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
            self.addObserver(slicer.mrmlScene, slicer.vtkMRMLScene.NodeAddedEvent, self.onNodeAdded)

            self.updateGUIFromParameterNode(None, None)


#
# ContactDetectorLogic
#


class ContactDetectorLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return ContactDetectorParameterNode(super().getParameterNode())
    
    def curve_fitting(self,
            inputCT: vtkMRMLScalarVolumeNode,
            electrodes: list[Electrode],
            sphere_radius_mm: float,
            blob_size_sigma: float,
            contact_diameter_mm: float,
            contact_length_mm: float,
            contact_gap_mm: float,
            show_gaussian_balls: bool):
        # import or install dependencies
        try:
            from sklearn.decomposition import PCA
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay("This module requires 'scikit-learn' Python package. Click OK to install it now."):
                slicer.util.pip_install("scikit-learn")
                from sklearn.decomposition import PCA

        # prepare data array
        ct_array = slicer.util.arrayFromVolume(inputCT)

        # precompute gaussian ball
        sigma_mm = contact_diameter_mm
        sigma_ijk = sigma_mm / np.array(inputCT.GetSpacing())[::-1]
        ball_shape = np.ceil(blob_size_sigma * sigma_ijk).astype(int) # blob_size_sigma defines size of the gaussian ball in sigmas

        ball_shape += ball_shape % 2 == 0 # make sure the shape is odd

        ball_radius = ball_shape // 2
        gaussian_ball = self.gaussian_ball(ball_shape, sigma_ijk)

        # prepare markups
        markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        markupsNode.SetName("Electrodes")

        for electrode in electrodes:
            slicer.app.processEvents()
            
            # find the direction of the maximum variance
            pca = PCA(n_components=1)
            pca.fit(electrode.gmm_segmentation_indices_ijk)
            pca_v_ijk = pca.components_[0]
            pca_centroid_ijk = pca.mean_

            # check pca_v_ijk direction is pointing outside the skull
            if np.dot(electrode.tip_ijk - electrode.entry_point_ijk, pca_v_ijk) > 0:
                pca_v_ijk = -pca_v_ijk

            # project the electrode points to the direction of the maximum variance
            s = (electrode.gmm_segmentation_indices_ijk - pca_centroid_ijk) @ pca_v_ijk

            # prepare weights
            intensities = ct_array[electrode.gmm_segmentation_indices_ijk[:, 0], electrode.gmm_segmentation_indices_ijk[:, 1], electrode.gmm_segmentation_indices_ijk[:, 2]]

            # fit a 5th order polynomial
            coeffs_x = np.polyfit(s, electrode.gmm_segmentation_indices_ijk[:, 0], 5, w=intensities)
            coeffs_y = np.polyfit(s, electrode.gmm_segmentation_indices_ijk[:, 1], 5, w=intensities)
            coeffs_z = np.polyfit(s, electrode.gmm_segmentation_indices_ijk[:, 2], 5, w=intensities)

            # extend projection of the points
            extend_mm = 2*(contact_length_mm + contact_gap_mm)
            extend_ijk = extend_mm / np.linalg.norm(pca_v_ijk * inputCT.GetSpacing()[::-1])

            # electrode.length_mm + backwards_mm + sphere around the bolt / 0.1 mm resolution
            num_points = np.ceil((electrode.length_mm + extend_mm + 2*sphere_radius_mm) / 0.1).astype(int)
            X = np.linspace(s.min()-extend_ijk, s.max(), num_points)
            x_fit = np.polyval(coeffs_x, X)
            y_fit = np.polyval(coeffs_y, X)
            z_fit = np.polyval(coeffs_z, X)

            electrode.curve_points = np.column_stack((x_fit, y_fit, z_fit))

            # compute distance between points on the curve
            diffs = np.diff(electrode.curve_points * inputCT.GetSpacing()[::-1], axis=0)
            curve_distances_between_points = np.linalg.norm(diffs, axis=1)
            electrode.curve_cumulative_distances = np.cumsum(curve_distances_between_points)

            # get number of points between the first and the fifth contact
            selected_points = self.select_contact_points(electrode.curve_points, electrode.curve_cumulative_distances, 0, electrode.n_contacts, contact_length_mm, contact_gap_mm)
            fifth_contact_point = selected_points[4]
            n = np.where(electrode.curve_points == fifth_contact_point)[0][0]

            best_fit = -np.inf
            best_gaussian = None
            for offset in range(n):
                slicer.app.processEvents()

                # select points from the fitted curve
                selected_points = self.select_contact_points(electrode.curve_points, electrode.curve_cumulative_distances, offset, electrode.n_contacts, contact_length_mm, contact_gap_mm)
                gaussian_balls_volume = np.zeros(ct_array.shape)

                # generate gaussian balls
                for point in selected_points:
                    # add precomputed ball to particular location
                    min_voxel = np.round(point - ball_radius).astype(int)
                    max_voxel = np.round(point + ball_radius + 1).astype(int)

                    # clip to image boundaries
                    min_vol = np.maximum(min_voxel, 0)
                    max_vol = np.minimum(max_voxel, ct_array.shape)

                    ball_start = min_vol - min_voxel
                    ball_end = max_vol - min_voxel

                    gaussian_balls_volume[
                        min_vol[0]:max_vol[0],
                        min_vol[1]:max_vol[1],
                        min_vol[2]:max_vol[2]] += gaussian_ball[ball_start[0]:ball_end[0],
                                                                ball_start[1]:ball_end[1],
                                                                ball_start[2]:ball_end[2]]

                # find absolute bounding box
                min_voxel = np.round(selected_points.min(axis=0) - ball_radius).astype(int)
                max_voxel = np.round(selected_points.max(axis=0) + ball_radius + 1).astype(int)

                # bounding box
                bbox_min = np.maximum(min_voxel, 0)
                bbox_max = np.minimum(max_voxel, gaussian_balls_volume.shape)

                # crop region
                gbv_crop = gaussian_balls_volume[bbox_min[0]:bbox_max[0],
                                                 bbox_min[1]:bbox_max[1],
                                                 bbox_min[2]:bbox_max[2]]

                ct_crop = ct_array[bbox_min[0]:bbox_max[0],
                                   bbox_min[1]:bbox_max[1],
                                   bbox_min[2]:bbox_max[2]]

                # compute correlation
                mask = gbv_crop != 0
                correlation = np.corrcoef(gbv_crop[mask], ct_crop[mask])[0, 1]

                if correlation > best_fit:
                    best_fit = correlation
                    points_best_fit = selected_points
                    electrode.curve_points_offset = offset
                    best_gaussian = gaussian_balls_volume

            if show_gaussian_balls:
                volumeNode = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, inputCT, f"Gaussian balls {electrode.label_prefix}")
                slicer.util.updateVolumeFromArray(volumeNode, best_gaussian.astype(np.float32))

            # plot selected points
            for i, point in enumerate(points_best_fit):
                points_best_fit[i] = self.IJK_to_RAS(point[::-1], inputCT)
                markupsNode.AddControlPoint(points_best_fit[i], f"{electrode.label_prefix}{i+1}")

        return markupsNode

    def eletrode_segmentation(self,
                              inputCT: vtkMRMLScalarVolumeNode, 
                              brainMask: vtkMRMLSegmentationNode,
                              electrodes: list[Electrode],
                              metal_threshold: float,
                              add_segmentation: bool):
        # import or install dependencies
        try:
            from sklearn.mixture import GaussianMixture
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay("This module requires 'scikit-learn' Python package. Click OK to install it now."):
                slicer.util.pip_install("scikit-learn")
                from sklearn.mixture import GaussianMixture

        # prepare copy of the data array
        ct_array = slicer.util.arrayFromVolume(inputCT).copy()
        brain_mask_array = slicer.util.arrayFromSegmentBinaryLabelmap(brainMask, brainMask.GetSegmentation().GetSegmentIDs()[0], inputCT)

        # apply brain mask and threshold
        ct_array[(brain_mask_array == 0) | (ct_array < metal_threshold)] = 0

        # include bolt area
        for electrode in electrodes:
            slicer.app.processEvents()
            i,j,k = electrode.bolt_segmentation_indices_ijk.T
            ct_array[i,j,k] = slicer.util.arrayFromVolume(inputCT)[i,j,k]

        # prepare GMM
        midpoints = []
        covariances = []
        for electrode in electrodes:
            slicer.app.processEvents()
            midpoint = (electrode.tip_ijk + electrode.entry_point_ijk) / 2
            midpoints.append(midpoint)
            
            X = np.vstack((electrode.tip_ijk, electrode.entry_point_ijk))
            covariance = np.cov(X, rowvar=False)
            epsilon = 0.01 * np.max(np.diag(covariance))
            covariances.append(np.linalg.inv(covariance + epsilon * np.eye(3)))

        # run GMM
        gmm = GaussianMixture(n_components=len(electrodes),
                              covariance_type="full",
                              means_init=np.array(midpoints),
                              precisions_init=np.array(covariances),
                              weights_init=np.full(len(electrodes), 1/len(electrodes)))

        points = np.array(np.nonzero(ct_array)).T
        gmm.fit(points)
        labels = gmm.predict(points)
        
        if add_segmentation:
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segmentationNode.SetName("Electrodes")
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputCT)
            segmentationNode.CreateDefaultDisplayNodes()
       
        # assign gmm labels
        for i, electrode in enumerate(electrodes):
            slicer.app.processEvents()
            electrode.gmm_segmentation_indices_ijk = points[labels == i]
            
            if add_segmentation:
                # prepare array with electrode segmentation
                gmm_volume_array = np.zeros_like(ct_array, dtype=np.int8)
                i,j,k = electrode.gmm_segmentation_indices_ijk.T
                gmm_volume_array[i,j,k] = 1

                segmentId = segmentationNode.GetSegmentation().AddEmptySegment()
                segmentationNode.GetSegmentation().GetSegment(segmentId).SetName(f"Electrode {electrode.label_prefix}")
                slicer.util.updateSegmentBinaryLabelmapFromArray(gmm_volume_array, segmentationNode, segmentId)

    def bolt_axis_estimation(self,
                             inputCT: vtkMRMLScalarVolumeNode,
                             electrodes: list[Electrode],
                             add_line: bool) -> None:
        # import or install dependencies
        try:
            from sklearn.decomposition import PCA
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay("This module requires 'scikit-learn' Python package. Click OK to install it now."):
                slicer.util.pip_install("scikit-learn")
                from sklearn.decomposition import PCA

        # prepare data array
        ct_array = slicer.util.arrayFromVolume(inputCT)

        for electrode in electrodes:
            slicer.app.processEvents()
            
            # find the best fit line for the electrode
            pca = PCA(n_components=1)
            pca.fit(electrode.bolt_segmentation_indices_ijk)
            pca_v_ijk = pca.components_[0]
            pca_centroid_ijk = pca.mean_

            # check pca_v_ijk direction is pointing inside the skull
            ct_image_center_ijk = np.array(ct_array.shape) / 2
            vector_to_center = ct_image_center_ijk - pca_centroid_ijk
            if np.dot(vector_to_center, pca_v_ijk) < 0:
                pca_v_ijk = -pca_v_ijk

            # entry point of the electrode is a projection of the tip of the bolt to the bolt axis
            w = self.RAS_to_IJK(electrode.bolt_tip_ras, inputCT)[::-1] - pca_centroid_ijk
            t = np.dot(w, pca_v_ijk)
            electrode.entry_point_ijk = pca_centroid_ijk + t * pca_v_ijk

            # compute offset of the electrode tip in direction of the bolt
            voxel_spacing = inputCT.GetSpacing()[::-1]
            norm_factor = np.linalg.norm(voxel_spacing * pca_v_ijk) # length of the PCA unit vector in mm
            offset_ijk = (pca_v_ijk * electrode.length_mm) / norm_factor

            electrode.tip_ijk = electrode.entry_point_ijk + offset_ijk

            if add_line:
                lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
                lineNode.SetName(f"Electrode {electrode.label_prefix}")
                lineNode.SetLineStartPosition(self.IJK_to_RAS(electrode.entry_point_ijk[::-1], inputCT)) # flip coordinates
                lineNode.SetLineEndPosition(self.IJK_to_RAS(electrode.tip_ijk[::-1], inputCT))

    def bolt_segmentation(self,
                          inputCT: vtkMRMLScalarVolumeNode,
                          electrodes: list[Electrode],
                          sphere_radius_mm: float,
                          metal_threshold: float,
                          add_segmentation: bool) -> None:
        # import or install dependencies
        try:
            import skimage
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay("This module requires 'scikit-image' Python package. Click OK to install it now."):
                slicer.util.pip_install("scikit-image")
                import skimage

        # prepare data array
        ct_array = slicer.util.arrayFromVolume(inputCT)

        # create a spherical mask
        spherical_mask = self.create_spherical_mask(np.array(inputCT.GetSpacing())[::-1], sphere_radius_mm)

        if add_segmentation:
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segmentationNode.SetName("Bolts")
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputCT)
            segmentationNode.CreateDefaultDisplayNodes()
        
        for electrode in electrodes:
            slicer.app.processEvents()

            bolt_tip_ijk = self.RAS_to_IJK(electrode.bolt_tip_ras, inputCT)[::-1] # flip coordinates to match flipped array

            sphere_center = np.array(spherical_mask.shape) // 2
            min_voxel = np.round(bolt_tip_ijk - sphere_center).astype(int)
            max_voxel = np.round(bolt_tip_ijk + sphere_center + 1).astype(int)

            # clip to image boundaries
            min_vol = np.maximum(min_voxel, 0)
            max_vol = np.minimum(max_voxel, ct_array.shape)

            mask_start = min_vol - min_voxel
            mask_end = max_vol - min_voxel

            # apply sphere
            bolt_segmentation = spherical_mask[mask_start[0]:mask_end[0],
                                               mask_start[1]:mask_end[1],
                                               mask_start[2]:mask_end[2]].copy()
            
            # threshold metal
            ct_crop = ct_array[min_vol[0]:max_vol[0],
                               min_vol[1]:max_vol[1],
                               min_vol[2]:max_vol[2]]
            bolt_segmentation[ct_crop < metal_threshold] = 0

            # find the largest connected component (full connectivity)
            labels = skimage.measure.label(bolt_segmentation)
            counts = np.bincount(labels.ravel())
            largest_label = np.argmax(counts[1:]) + 1
            bolt_segmentation[labels != largest_label] = 0

            # save mask
            electrode.bolt_segmentation_indices_ijk = np.array(np.nonzero(bolt_segmentation)).T + min_vol

            if add_segmentation:
                bolt_segm = np.zeros(ct_array.shape, dtype=np.uint8)
                bolt_segm[min_vol[0]:max_vol[0],
                          min_vol[1]:max_vol[1],
                          min_vol[2]:max_vol[2]] = bolt_segmentation

                segmentId = segmentationNode.GetSegmentation().AddEmptySegment()
                segmentationNode.GetSegmentation().GetSegment(segmentId).SetName(f"Bolt {electrode.label_prefix}")
                slicer.util.updateSegmentBinaryLabelmapFromArray(bolt_segm, segmentationNode, segmentId)

    def gaussian_ball(self,
                      shape: np.array,
                      sigma_ijk: np.array) -> np.array:
        x, y, z = np.indices(shape)
        center = shape // 2
        
        gauss = np.exp(
            -((x - center[0])**2 / (2 * sigma_ijk[0]**2)
              + (y - center[1])**2 / (2 * sigma_ijk[1]**2)
              + (z - center[2])**2 / (2 * sigma_ijk[2]**2)
              ))

        return gauss

    def select_contact_points(self,
                              points: np.array,
                              cumulative_distances: np.array,
                              n_offset: int,
                              n_contacts: int,
                              contact_length_mm: float,
                              contact_gap_mm: float,
                              step: int = 0) -> np.array:
        offset_distance = cumulative_distances[n_offset]
        target_distances = offset_distance + np.arange(n_contacts) * (contact_length_mm + contact_gap_mm) + step * (contact_length_mm + contact_gap_mm)

        chosen_idx = []
        for t in target_distances:
            # insertion point where cumulative >= t
            idx = np.searchsorted(cumulative_distances, t)

            # choose nearer of the two neighbors
            below = cumulative_distances[idx - 1]
            above = cumulative_distances[idx]
            best = idx if abs(above - t) < abs(t - below) else idx - 1

            chosen_idx.append(best)

        return points[chosen_idx]

    def create_spherical_mask(self,
                              spacing: np.array,
                              radius_mm: float = 10) -> np.array:
        radius_voxels = np.floor(radius_mm / spacing).astype(int)
        shape = 2 * radius_voxels + 1

        # 0 is the center
        x = np.arange(shape[0]) - radius_voxels[0]
        y = np.arange(shape[1]) - radius_voxels[1]
        z = np.arange(shape[2]) - radius_voxels[2]

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        dist_mm = np.sqrt((xx*spacing[0])**2 + (yy*spacing[1])**2 + (zz*spacing[2])**2)
        sphere_mask = dist_mm <= radius_mm

        return sphere_mask

    def RAS_to_IJK(self,
                 point_RAS: np.array,
                 volumeNode: vtkMRMLScalarVolumeNode) -> np.array:
        volumeRasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
        point_Ijk = volumeRasToIjk.MultiplyPoint(np.append(point_RAS,1.0))
        return np.array(point_Ijk[:3])
    
    def IJK_to_RAS(self,
                 point_IJK: np.array,
                 volumeNode: vtkMRMLScalarVolumeNode) -> np.array:
        volumeIjkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(volumeIjkToRas)
        point_Ras = volumeIjkToRas.MultiplyPoint(np.append(point_IJK,1.0))
        return np.array(point_Ras[:3])


#
# ContactDetectorTest
#


class ContactDetectorTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_ContactDetector1()

    def test_ContactDetector1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("ContactDetector1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = ContactDetectorLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
