# Features
- [ ] add registration
- [ ] skull stripping
- [x] automatic installation of python packages
- [X] contact manual shifting
- [x] use cropped array around the balls for the correlation for faster computation

# Visualization
- [x] debug checkbox to enable visualization
- [ ] compute distance between gmm centroids and throw warning if one distance is out of tolerance

# Debugging
- [ ] check centers of the spheres and of the gaussinan balls
- [ ] test edge cases (bolt spheres around the edge of the image, gaussian balls around the edge of the image)
- [ ] check algorithm on non standard spacing
- [ ] fix memory leakage after deleting nodes from slicer

# Finalization
- [ ] remove step-by-step, add event on run all click
- [ ] evaluate differences against GT