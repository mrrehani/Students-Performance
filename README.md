# CNN-Wheres-Waldo

![Contributors](https://img.shields.io/badge/Contributors-Michael%20Rehani-brightgreen)
![Licence](https://img.shields.io/github/license/mrrehani/Students-Performance)
![Size](https://img.shields.io/github/repo-size/mrrehani/Students-Performance)
<br>
<a href="https://www.linkedin.com/in/michael-rehani/">
<img alt="Connect with me on LinkedIn!">
</a>

## Description
In this project, I focused on predicting student test performance based on a range of academic and socioeconomic factors. I evaluated each method individually by performing principal component analysis (using the number of components that yielded the highest accuracy), and generated a final classification accuracy per method. To test if my results could be improved, I performed a combinatorial analysis of all methods to generate groups that were ensembled together and then voted on to create a composite answer (individual method contribution was weighted by accuracy, and a final score was normalized to 0 to 1; final score > 0.6 was deemed to be a 1, else 0). In doing this, I improved the accuracy of my model from ~72% for the best performing individual method to ~80% for the optimal combination of methods. 

### Technologies Used:
- Sklearn
- ImageDataGenerator
- Matplotlib

### Classification Algorithms Used:
- Decision Trees
- Gaussian Naive Bayes
- K-Nearest Neighbor
- Support Vector Machines
- Random Forest
- Logistic Regression

### Instructions
- Clone the repository
- Open the notebook file, making sure that the 'StudentsPerformance.csv' file is still in the same directory
- Run the all the cells

### Credits
The original source of the data can be found [here](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).
