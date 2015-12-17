#!/bin/bash
echo Running ${1} > /proj/SMACK/mlproj/${2}
python3 cuisineClassifier.py ${1} &>> /proj/SMACK/mlproj/${2}
