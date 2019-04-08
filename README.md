# ai-art-creation
Creation of new images using genetic algorithms

## Structure
1. Code - contains 4 python scripts. 
        Each of them generate output using arcs, white-and-black arcs, letters or triangles.
        The type of object used to generate ouput depends on the name of the script
1. Fonts - contains fonts used to generate output using letters
1. Images - original images which should be processed
1. Populations - some kind of backup where populations 
        obtained during execution are written as numpy files.
        They help to continue if program crashes or will be stopped
1. Results - folder which is used to write output in
1. Summary - contains code and results which are described in the report
1. Tests - contains tests of different approaches and their resultst

## Scripts
1. **arcs.py** - create output image from color arcs
1. **arcs_engraving.py** - create output image from grayscale arcs
1. **letters.py** - create output image from color letters
1. **triangles.py** - create output image from color triangles

## Libraries used
All used libraries are mentioned into **requirements.txt** file

## Running
1. Install pip for your Python 3 interpreter
1. Using command `pip install -r requirements.txt` install all necessary libraries
1. Add input image in the JPEG format into folder **images** with name 
        **input.jpg**
1. Run a script from **code** directory which produce the output 
        with objects you want to use (arcs.py, acrs_engraving.py, letters.py, triangles.py)
1. Wait until program terminates
1. The output image will be located in the **results** folder 

