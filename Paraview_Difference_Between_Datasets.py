#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Legacy VTK Reader'
step01_t0000000vtk = LegacyVTKReader(FileNames=['/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_02/output/step01_t0000000.vtk'])

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1177, 1025]

# show data in view
step01_t0000000vtkDisplay = Show(step01_t0000000vtk, renderView1)
# trace defaults for the display properties.
step01_t0000000vtkDisplay.ColorArrayName = [None, '']
step01_t0000000vtkDisplay.OSPRayScaleArray = 'displacement'
step01_t0000000vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
step01_t0000000vtkDisplay.GlyphType = 'Arrow'
step01_t0000000vtkDisplay.ScalarOpacityUnitDistance = 13465.362619693818
step01_t0000000vtkDisplay.SetScaleArray = [None, '']
step01_t0000000vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
step01_t0000000vtkDisplay.OpacityArray = [None, '']
step01_t0000000vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [-400.0, 0.0, 10000.0]
renderView1.CameraFocalPoint = [-400.0, 0.0, 0.0]

# create a new 'Legacy VTK Reader'
step01_t0000100vtk = LegacyVTKReader(FileNames=['/Users/josimar/Documents/Work/Projects/SlowEarthquakes/Modeling/PyLith/Runs/Calibration/2D/version_02/output/step01_t0000100.vtk'])

#Attempting to create the difference between the two datasets
elev0 = step01_t0000000vtk.PointData['displacement']
elev1 = step01_t0000100vtk.PointData['displacement']
output.PointData.append(elev1 - elev0, 'difference')


# show data in view
step01_t0000100vtkDisplay = Show(step01_t0000100vtk, renderView1)
# trace defaults for the display properties.
step01_t0000100vtkDisplay.ColorArrayName = [None, '']
step01_t0000100vtkDisplay.OSPRayScaleArray = 'displacement'
step01_t0000100vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
step01_t0000100vtkDisplay.GlyphType = 'Arrow'
step01_t0000100vtkDisplay.ScalarOpacityUnitDistance = 13465.362619693818
step01_t0000100vtkDisplay.SetScaleArray = [None, '']
step01_t0000100vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
step01_t0000100vtkDisplay.OpacityArray = [None, '']
step01_t0000100vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'

# hide data in view
Hide(step01_t0000000vtk, renderView1)

# set scalar coloring
ColorBy(step01_t0000100vtkDisplay, ('POINTS', 'displacement'))

# rescale color and/or opacity maps used to include current data range
step01_t0000100vtkDisplay.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
step01_t0000100vtkDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'displacement'
displacementLUT = GetColorTransferFunction('displacement')

# get opacity transfer function/opacity map for 'displacement'
displacementPWF = GetOpacityTransferFunction('displacement')

# hide data in view
Hide(step01_t0000100vtk, renderView1)

# set active source
SetActiveSource(step01_t0000100vtk)

# show data in view
step01_t0000100vtkDisplay = Show(step01_t0000100vtk, renderView1)

# show color bar/color legend
step01_t0000100vtkDisplay.SetScalarBarVisibility(renderView1, True)

# reset view to fit data
renderView1.ResetCamera()

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [-400.0, 0.0, 578928.7217577639]
renderView1.CameraFocalPoint = [-400.0, 0.0, 0.0]
renderView1.CameraParallelScale = 149837.77894776737

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).