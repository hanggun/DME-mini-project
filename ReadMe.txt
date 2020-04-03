#####
CODES
#####
./Codes: Contains all the code files.

./Codes/label.csv: This file is manually converted from 'tissues.html'.

./Codes/ProcessInputData.py: To process 'I2000.html' and save as 'colonCancerData.csv'.

./Codes/ProcessNamesData.py: To process 'names.html' and save as 'colonGeneDescriptions.csv'.

./Codes/Descriptive analysis.ipynb: The code file that performs EDA, and shown in report section 2.

./Codes/heatmap.py: The code file to draw the heat map that is shown in report section 3.

./Codes/ColonTissueLib.py: Contains the codes for feature selection, and other utility helper functions. Mainly used by 'ColonTissueClassifiers.py' and 'EvaluateColonTissueClassifiers.py'.

./Codes/ColonTissueClassifiers.py: The code file that performs model training using LOOCV and Bayesian optimization. All optimized models are saved into pickle files in the 'Models' directory.

./Codes/EvaluateColonTissueClassifiers.py: The code file that evaluates the trained models' LOOCV accuracy. It also evaluates the test performance of 3 selected classifiers based on LOOCV accuracy. All the figures are saved in the 'Figures' directory.

#####
LATEX
#####
./LatexFiles: Contains all the Latex files used to generate our report.

#####
REPORT
#####
./DME_group11_project_report.pdf: This is our report file for the mini project.