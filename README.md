# mml_dinolite_vision
Image-based analysis tools for Dino-Lite microscope data used in MML spindle test-touch characterization. This repository focuses on detecting, grouping, and measuring cut features from microscope images to estimate test-touch lengths for filters and lens production 


`dinolite.py` is what's used in the backend to do a variety of tasks with the Dinolite Digital Microscope. I.e., preview the camera, query its current state (AMR, fov, EDOF, etc), and capture an image when we're ready.

`capture_image.py` will take the photo when we're ready. This is useful for when we control the Aerotech automation1 hardware to go to a specific set of coordinates, and once we know we're there, we then have the Dinolite capture the image. 

`detect_test_touches.py` will have the host of functions we need to call to analyze the image to output the measured test touch value
