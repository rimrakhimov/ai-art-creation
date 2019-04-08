# ai-art-creation
Creation of new images using genetic algorithms

<h3>Structure</h3>
<ol>
    <li>Code - contains 4 python scripts. 
        Each of them generate output using arcs, white-and-black arcs, letters or triangles.
        The type of object used to generate ouput depends on the name of the script</li>   
    <li>Fonts - contains fonts used to generate output using letters</li>
    <li>Images - original images which should be processed</li>
    <li>Populations - some kind of backup where populations 
        obtained during execution are written as numpy files.
        They help to continue if program crashes or will be stopped</li>
    <li>Results - folder which is used to write output in</li>
    <li>Summary - contains code and results which are described in the report</li>
    <li>Tests - contains tests of different approaches and their resultst</li>
</ol>

<h3>Scripts</h3>
<ol>
    <li><strong>arcs.py</strong> - create output image from color arcs</li>
    <li><strong>arcs_engraving.py</strong> - create output image from grayscale arcs</li>
    <li><strong>letters.py</strong> - create output image from color letters</li>
    <li><strong>triangles.py</strong> - create output image from color triangles</li>
</ol>

<h3>Running</h3>
<ol>
    <li>Add input image in the JPEG format into folder <strong>images</strong> with name 
        <strong>input.jpg</strong></li>
    <li>Run a script from <strong>code</strong> folder which produce the output 
        with objects you want to use (arcs.py, acrs_engraving.py, letters.py, triangles.py)</li>
    <li>Wait until program terminates</li>
    <li>The output image will be located in the <strong>results</strong> folder</li> 
         
</ol>

