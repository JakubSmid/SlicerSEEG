# Features
- [x] add registration
- [x] skull stripping
- [x] automatic installation of python packages
- [X] contact manual shifting
- [x] use cropped array around the balls for the correlation for faster computation
- [ ] CT bone reconstruction
- [x] auto load input based on input file name
- [x] fiducial placement
- [x] use tip of the electrode instead of the linear approximation
- [x] add micro step while electrode shifting
- [x] new fiducial naming A-no, B-no
- [x] disabled buttons tooltip hint

# Visualization
- [x] debug checkbox to enable visualization
- [ ] compute distance between gmm centroids and throw warning if one distance is out of tolerance

# Debugging
- [x] check centers of the spheres and of the gaussinan balls
- [x] test edge cases (bolt spheres around the edge of the image, gaussian balls around the edge of the image)
- [x] check algorithm on non standard spacing
- [x] fix memory leakage after deleting nodes from slicer
- [x] warning if there is no bolt around the input fiducial
- [x] fix error when extending the curve outside of the image / moving the electrode outside of the image
- [x] critical error when shifting contacts after renaming estimated contacts
- [ ] cannot make window smaller due to the minimum size of the widgets => make Inputs buttons in vertical layout?
- [ ] change estimation of the electrode axis orientation

# Finalization
- [x] add parameters into the parameter node and check scene saving
- [x] remove step-by-step, add event on run all click
- [ ] create tooltip help
- [ ] add module description, help and acknowledgement
- [ ] icon

# Testing
- [x] prepare dataset
- [x] apply dummy affine transform to the T1
- [x] estimate bolt tip as extrapolation of the first and last contact and apply additional random translation away from the axis
- [ ] evaluation script
- [x] batch loading (as a test function?)

# Known issues
- [x] while fitting the electrode with less than 5 contacts, out of range warning is raised