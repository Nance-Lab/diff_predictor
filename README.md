## diff_predictor



<p>
	<img src="https://avatars0.githubusercontent.com/u/64927580?s=200&v=4" width="10%" align="left">
</p>



# diff_predictor
Diff_predictor is a package to work in tandem with diff_classifier (https://github.com/Nance-Lab/diff_classifier). 

### Description:
This package contains tools for prediction and analysis of multiple particle tracking data of nanoparticles taken from biological tissue imaging.

### Organization of the project:
The project has the following structure:

```
|- diff_predictor/
   |- LICENSE
   |- README.md
   |- requirements.txt
   |- setup.py
   |- diff_predictor/
	  |- tests/
		 |- test_data_process.py
		 |- test_dataio.py
		 |- test_eval.py
		 |- test_predxgboost.py
	  |- __init__.py
	  |- data_process.py
	  |- dataio.py
	  |- eval.py
	  |- predxgboost.py
	  |- temporal.py
	  |- version.py
	  |- notebooks/
		 |- ...

```
### Installation
#### Pip Install
Currently, diff_predictor is not on PyPi, but the next version of diff_predictor will be uploaded to PyPi.

#### Cloning
Users can clone a copy of diff_predictor with the command

```git clone https://github.com/dash2927/diff_predictor.git```

Running the setup file will install needed dependencies:

```python3 setup.py develop```

You can install diff_predictor from the Github repository. This will install diff_predictor and its Python dependencies.

#### GPU Computing
Default installation uses CPU computing. It is recomended to run xgboost and tensorflow commands using gpu enabled computing. To do this see:

https://www.tensorflow.org/install/gpu


### Analysis Notebooks
Data analysis was performed male Sprague-Dawley (SD) rat pups at varying ages, depending on the specific study. These analysis notebooks can be found in the diff_predictor/notebooks folder of the package. Individual slices were plated on 30-mm cell culture inserts in non-treated 6-well plates. Prior to plating, 6-well plates were filled with 1 mL SCM. Slices were incubated in sterile conditions at 37°C and 5% CO2. Predictive analysis tested for pup age and for brain region.

All MPT studies were performed within 24 h of slice preparation. Slices were imaged in a temperature-controlled incubation chamber maintained at 37°C, 5% CO2, and 80% humidity. 30 minutes (min) prior to video acquisition, 40nm polystyrene nanoparticles conjugated with poly(ethylene-glycol) (PS-PEG) were diluted in 1x phosphate-buffered saline (PBS) to a concentration of ~0.014%. Nanoparticles were injected into each slice using a 10 µL glass syringe (model 701, cemented needle, 26-gauge, Hamilton Company, Reno, NV). Videos were collected at 33 frames-per-second and 100x magnification for 651 frames via fluorescent microscopy using a cMOS camera (Hamamatsu Photonics, Bridgewater, NJ) mounted on a confocal microscope. Nanoparticle trajectories and trajectory mean square displacements (MSDs) were calculated via diff_classifier (https://github.com/Nance-Lab/diff_classifier), a Python package developed within Nance Lab.

### References
1. Curtis, C., A. Rokem, and E. Nance, diff_classifier: Parallelization of multi-particle tracking video analyses. Journal of open source software, 2019. 4(36): p. 989.
2. Shapley, L.S., A value for n-person games. Contributions to the Theory of Games, 1953. 2(28): p. 307-317.
