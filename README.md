# Features
- [ ] add registration
- [ ] skull stripping
- [x] automatic installation of python packages
- [X] contact manual shifting
- [x] use cropped array around the balls for the correlation for faster computation
- [ ] CT bone reconstruction
- [ ] auto load input based on input file name
- [ ] use tip of the electrode instead of the linear approximation
- [ ] add micro step while electrode shifting

# Visualization
- [x] debug checkbox to enable visualization
- [ ] compute distance between gmm centroids and throw warning if one distance is out of tolerance

# Debugging
- [x] check centers of the spheres and of the gaussinan balls
- [x] test edge cases (bolt spheres around the edge of the image, gaussian balls around the edge of the image)
- [x] check algorithm on non standard spacing
- [x] fix memory leakage after deleting nodes from slicer
- [ ] warning if there is no bolt around the input fiducial
- [ ] fix error when extending the curve outside of the image / moving the electrode outside of the image

# Finalization
- [ ] remove step-by-step, add event on run all click
- [ ] create tooltip help
- [ ] add module description, help and acknowledgement
- [ ] evaluate differences against GT