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
import skimage
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

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

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode
    inputCT: vtkMRMLScalarVolumeNode
    brainMask: vtkMRMLSegmentationNode
    boltFiducials: vtkMRMLMarkupsFiducialNode


class Electrode():
    def __init__(self, bolt_tip_ras, label):
        self.bolt_tip_ras = np.array(bolt_tip_ras)
        self.label = label
        self.label_prefix, self.n_contacts = Electrode.split_label(label)
        self.length_mm = 2*self.n_contacts + 1.5 * (self.n_contacts - 1)
        
        self.bolt_mask_array = None
        self.tip_ijk = None
        self.entry_point_ijk = None
        self.gmm_label = None
    
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

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ContactDetectorLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        # self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        # self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        # self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.buttonBoltSegmentation.connect("clicked(bool)", self.onBoltSegmentationButton)
        self.ui.buttonBoltAxisEst.connect("clicked(bool)", self.onBoltAxisEstButton)
        self.ui.buttonElectrodeSegmentation.connect("clicked(bool)", self.onElectrodeSegmentationButton)
        self.ui.buttonCurveFitting.connect("clicked(bool)", self.onCurveFittingButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def onCurveFittingButton(self):
        self.logic.curve_fitting(self._parameterNode.inputCT, self.electrode_labels_volume_array, self.electrodes)

    def onBoltAxisEstButton(self):
        self.logic.bolt_axis_estimation(self._parameterNode.inputCT, self.electrodes)

    def onBoltSegmentationButton(self):
        # load electrodes
        self.electrodes: list[Electrode] = []
        for i in range(self._parameterNode.boltFiducials.GetNumberOfControlPoints()):
            bolt_tip_ras = [0, 0, 0]
            self._parameterNode.boltFiducials.GetNthControlPointPosition(i, bolt_tip_ras)
            self.electrodes.append(Electrode(bolt_tip_ras, self._parameterNode.boltFiducials.GetNthControlPointLabel(i)))

        self.logic.bolt_segmentation(self._parameterNode.inputCT, self.electrodes)

    def onElectrodeSegmentationButton(self):
        self.electrode_labels_volume_array = self.logic.eletrode_segmentation(self._parameterNode.inputCT, self._parameterNode.brainMask, self.electrodes)

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
            # self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.inputVolume:
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[ContactDetectorParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            # self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            # self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            # self._checkCanApply()

    # def _checkCanApply(self, caller=None, event=None) -> None:
    #     if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
    #         self.ui.applyButton.toolTip = _("Compute output volume")
    #         self.ui.applyButton.enabled = True
    #     else:
    #         self.ui.applyButton.toolTip = _("Select input and output volume nodes")
    #         self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


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
            labels_volume: np.ndarray,
            electrodes: list[Electrode]):
        progressbar = slicer.util.createProgressDialog()
        progressbar.setCancelButton(None)
        slicer.app.processEvents()

        # precompute gaussian ball
        sigma_mm = 0.8
        sigma_ijk = sigma_mm / np.array(inputCT.GetSpacing())[::-1]
        ball_shape = (3 * sigma_ijk).astype(int) # shape of 3 sigma
        
        # ensure odd shape
        if ball_shape[0] % 2 == 0:
            ball_shape[0] += 1
        if ball_shape[1] % 2 == 0:
            ball_shape[1] += 1
        if ball_shape[2] % 2 == 0:
            ball_shape[2] += 1

        ball_radius = ball_shape // 2
        gaussian_ball = self.gaussian_ball(ball_shape, sigma_ijk)

        for electrode in electrodes:
            electrode_points = np.array(np.nonzero(labels_volume == electrode.gmm_label)).T
            
            # find the direction of the maximum variance
            pca = PCA(n_components=1)
            pca.fit(electrode_points)
            pca_v_ijk = pca.components_[0]
            pca_centroid_ijk = pca.mean_

            # check pca_v_ijk direction is pointing outside the skull
            if np.dot(electrode.tip_ijk - electrode.entry_point_ijk, pca_v_ijk) > 0:
                pca_v_ijk = -pca_v_ijk

            # project the electrode points to the direction of the maximum variance
            s = (electrode_points - pca_centroid_ijk) @ pca_v_ijk

            # fit a 5th order polynomial
            coeffs_x = np.polyfit(s, electrode_points[:, 0], 5)
            coeffs_y = np.polyfit(s, electrode_points[:, 1], 5)
            coeffs_z = np.polyfit(s, electrode_points[:, 2], 5)

            X = np.linspace(s.min(), s.max(), np.ceil((electrode.length_mm + 20) / 0.1).astype(int)) # electrode.length_mm + sphere around bolt (20 mm)
            x_fit = np.polyval(coeffs_x, X)
            y_fit = np.polyval(coeffs_y, X)
            z_fit = np.polyval(coeffs_z, X)

            points_list = np.column_stack((x_fit, y_fit, z_fit))

            # compute distance between points on the curve
            diffs = np.diff(points_list * inputCT.GetSpacing()[::-1], axis=0)
            distances_list = np.linalg.norm(diffs, axis=1)

            # get number of points between first and second contact
            selected_points = self.select_contact_points(points_list, distances_list, 0, electrode.n_contacts)
            second_contact_point = selected_points[1]
            n = np.where(points_list == second_contact_point)[0][0]

            points_best_fit = None
            best_fit = 0
            for offset in range(n):
                progressbar.setValue((offset+1) / n * 100)
                slicer.app.processEvents()
                
                # select points from the fitted curve
                selected_points = self.select_contact_points(points_list, distances_list, offset, electrode.n_contacts)
                gaussian_balls_volume = np.zeros(labels_volume.shape)

                # generate gaussian balls
                for point in selected_points:
                    # add precomputed ball to particular location
                    min_voxel = np.round(point - ball_radius).astype(int)
                    max_voxel = np.round(point + ball_radius + 1).astype(int)

                    # clip to image boundaries
                    min_vol = np.maximum(min_voxel, 0)
                    max_vol = np.minimum(max_voxel, gaussian_balls_volume.shape)

                    mask_start = min_vol - min_voxel
                    mask_end = max_vol - min_voxel

                    gaussian_balls_volume[
                        min_vol[0]:max_vol[0],
                        min_vol[1]:max_vol[1],
                        min_vol[2]:max_vol[2]] += gaussian_ball[mask_start[0]:mask_end[0],
                                                                mask_start[1]:mask_end[1],
                                                                mask_start[2]:mask_end[2]]

                # compute correlation
                correlation = np.corrcoef(gaussian_balls_volume.flatten(), slicer.util.arrayFromVolume(inputCT).flatten())[0, 1]
                if correlation > best_fit:
                    best_fit = correlation
                    points_best_fit = selected_points

            # save volume with blobs
            gaussian_balls_volume = np.zeros(labels_volume.shape)
            for point in points_best_fit:
                    # add precomputed ball to particular location
                    min_voxel = np.round(point - ball_radius).astype(int)
                    max_voxel = np.round(point + ball_radius + 1).astype(int)

                    # clip to image boundaries
                    min_vol = np.maximum(min_voxel, 0)
                    max_vol = np.minimum(max_voxel, gaussian_balls_volume.shape)

                    mask_start = min_vol - min_voxel
                    mask_end = max_vol - min_voxel

                    gaussian_balls_volume[
                        min_vol[0]:max_vol[0],
                        min_vol[1]:max_vol[1],
                        min_vol[2]:max_vol[2]] += gaussian_ball[mask_start[0]:mask_end[0],
                                                                mask_start[1]:mask_end[1],
                                                                mask_start[2]:mask_end[2]]
            gaussNode = slicer.modules.volumes.logic().CloneVolume(inputCT, f"Gaussian balls {electrode.label_prefix}")
            slicer.util.updateVolumeFromArray(gaussNode, gaussian_balls_volume)

            # plot selected points
            for i, point in enumerate(points_best_fit):
                points_best_fit[i] = self.IJK_to_RAS(point[::-1], inputCT)

            curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
            slicer.util.updateMarkupsControlPointsFromArray(curveNode, points_best_fit)
            curveNode.SetName(f"Electrode {electrode.label_prefix} curve")

    def eletrode_segmentation(self,
            inputCT: vtkMRMLScalarVolumeNode,
            brainMask: vtkMRMLSegmentationNode,
            electrodes: list[Electrode]):
        progressbar = slicer.util.createProgressDialog()
        progressbar.setCancelButton(None)
        slicer.app.processEvents()

        # threshold
        volume_array = slicer.util.arrayFromVolume(inputCT).copy()
        volume_array[volume_array < 3000] = 0

        # get brain mask
        brain_mask_labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode") # convert segmentation to labelmap
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(brainMask, brain_mask_labelmap, inputCT)
        brain_mask_array = slicer.util.arrayFromVolume(brain_mask_labelmap)
        slicer.mrmlScene.RemoveNode(brain_mask_labelmap)

        # apply brain mask
        volume_array[brain_mask_array == 0] = 0

        # include bolt area
        for electrode in electrodes:
            volume_array[electrode.bolt_mask_array == 1] = slicer.util.arrayFromVolume(inputCT)[electrode.bolt_mask_array == 1]

        # prepare GMM
        midpoints = []
        covariances = []
        for electrode in electrodes:
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
        
        points = np.array(np.nonzero(volume_array)).T
        gmm.fit(points)
        labels = gmm.predict(points)

        # create a label volume
        labels_volume_array = np.ones_like(volume_array, dtype=np.int8) * -1
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        labels_volume_array[x, y, z] = labels
        
        # assign gmm labels
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName("Electrodes")
        for i, electrode in enumerate(electrodes):
            progressbar.setValue((i+1) / len(electrodes) * 100)
            slicer.app.processEvents()

            i,j,k = np.round(electrode.entry_point_ijk).astype(int)
            electrode.gmm_label = labels_volume_array[i,j,k]
            gmm_volume_array = (labels_volume_array == electrode.gmm_label).astype(np.int8)
            
            # create label node
            label_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
            slicer.util.updateVolumeFromArray(label_node, gmm_volume_array)

            ijkToRas = vtk.vtkMatrix4x4()
            inputCT.GetIJKToRASMatrix(ijkToRas)
            label_node.SetIJKToRASMatrix(ijkToRas)
            label_node.SetOrigin(inputCT.GetOrigin())

            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label_node, segmentationNode)
            seg_id = segmentationNode.GetSegmentation().GetSegmentIDs()[-1]
            segmentationNode.GetSegmentation().GetSegment(seg_id).SetName(f"Electrode {electrode.label_prefix}")
            slicer.mrmlScene.RemoveNode(label_node)
        return labels_volume_array

    def bolt_axis_estimation(self,
                             inputCT: vtkMRMLScalarVolumeNode,
                             electrodes: list[Electrode]) -> None:
        progressbar = slicer.util.createProgressDialog()
        progressbar.setCancelButton(None)
        slicer.app.processEvents()

        for i, electrode in enumerate(electrodes):
            progressbar.setValue((i+1) / len(electrodes) * 100)
            slicer.app.processEvents()

            # find the best fit line for the electrode
            pca = PCA(n_components=1)
            points = np.array(np.nonzero(electrode.bolt_mask_array)).T
            pca.fit(points)
            pca_v_ijk = pca.components_[0]
            pca_centroid_ijk = pca.mean_

            # check pca_v_ijk direction is pointing inside the skull
            ct_image_center_ijk = np.array(electrode.bolt_mask_array.shape) / 2
            vector_to_center = ct_image_center_ijk - pca_centroid_ijk
            if np.dot(vector_to_center, pca_v_ijk) < 0:
                pca_v_ijk = -pca_v_ijk

            # entry point of the electrode is a projection of the tip of the bolt to the bolt axis
            w = np.array(self.RAS_to_IJK(electrode.bolt_tip_ras, inputCT))[::-1] - np.array(pca_centroid_ijk)
            t = np.dot(w, pca_v_ijk)
            electrode.entry_point_ijk = np.array(pca_centroid_ijk) + t * pca_v_ijk

            # compute offset of the electrode tip in direction of the bolt
            bolt_axis_mm = inputCT.GetSpacing()[::-1] * pca_v_ijk
            bolt_vector_mm_unit = bolt_axis_mm / np.linalg.norm(bolt_axis_mm)
            offset_mm = bolt_vector_mm_unit * electrode.length_mm
            offset_ijk = offset_mm / inputCT.GetSpacing()[::-1]

            electrode.tip_ijk = electrode.entry_point_ijk + offset_ijk

            # add line to scene
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            lineNode.SetName(f"Electrode {electrode.label_prefix}")
            lineNode.SetLineStartPosition(self.IJK_to_RAS(electrode.entry_point_ijk[::-1], inputCT)) # flip coordinates
            lineNode.SetLineEndPosition(self.IJK_to_RAS(electrode.tip_ijk[::-1], inputCT))

    def bolt_segmentation(self,
                          inputCT: vtkMRMLScalarVolumeNode,
                          electrodes: list[Electrode]) -> None:
        progressbar = slicer.util.createProgressDialog()
        progressbar.setCancelButton(None)
        slicer.app.processEvents()
        
        # prepare data array
        ct_array = slicer.util.arrayFromVolume(inputCT)

        # create a sphere with radius 10 voxels
        spherical_mask = self.create_spherical_mask(np.array(inputCT.GetSpacing()))

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName("Bolts")
        for i, electrode in enumerate(electrodes):
            progressbar.setValue((i+1) / len(electrodes) * 100)
            slicer.app.processEvents()

            electrode_ijk = self.RAS_to_IJK(electrode.bolt_tip_ras, inputCT)[::-1] # flip coordinates to match flipped array

            sphere_center = np.array(spherical_mask.shape) // 2
            min_voxel = np.round(electrode_ijk - sphere_center).astype(int)
            max_voxel = np.round(electrode_ijk + sphere_center + 1).astype(int)

            # clip to image boundaries
            min_vol = np.maximum(min_voxel, 0)
            max_vol = np.minimum(max_voxel, ct_array.shape)
            
            mask_start = min_vol - min_voxel
            mask_end = mask_start + (max_vol - min_vol)

            # apply sphere
            ct_mask = np.zeros(ct_array.shape, dtype=np.uint8)
            ct_mask[min_vol[0]:max_vol[0],
                    min_vol[1]:max_vol[1],
                    min_vol[2]:max_vol[2]] = spherical_mask[mask_start[0]:mask_end[0],
                                                            mask_start[1]:mask_end[1],
                                                            mask_start[2]:mask_end[2]]
            
            # threshold metal
            ct_mask[ct_array < 3000] = 0

            # find the largest connected component (full connectivity)
            labels = skimage.measure.label(ct_mask)
            counts = np.bincount(labels.ravel())
            largest_label = np.argmax(counts[1:]) + 1
            ct_mask[labels != largest_label] = 0

            # add mask of the electrode
            electrode.bolt_mask_array = ct_mask

            # add segmented bolt
            label_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
            slicer.util.updateVolumeFromArray(label_node, ct_mask)

            ijkToRas = vtk.vtkMatrix4x4()
            inputCT.GetIJKToRASMatrix(ijkToRas)
            label_node.SetIJKToRASMatrix(ijkToRas)
            label_node.SetOrigin(inputCT.GetOrigin())

            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label_node, segmentationNode)
            seg_id = segmentationNode.GetSegmentation().GetSegmentIDs()[-1]
            segmentationNode.GetSegmentation().GetSegment(seg_id).SetName(f"Bolt {electrode.label_prefix}")
            slicer.mrmlScene.RemoveNode(label_node)

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
                              distances: np.array,
                              n_offset: int,
                              n_contacts: int) -> np.array:
        contacts = [points[n_offset]]

        distance = 0
        last_point = n_offset
        for i in range(last_point, len(distances)):
            distance += distances[i] # cumulative sum between points

            # add contact if distance is 3.5 mm
            if distance >= 3.5:
                contacts.append(points[i+1])
                distance = 0
                last_point = i+1

                # break if we have all contacts
                if len(contacts) == n_contacts:
                    break

        return np.array(contacts)

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

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


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
