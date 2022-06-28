"""
This module compares several classification algorithms and determines the
optimal ensemble, creating several graphics to display results along the way.
"""
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt
import itertools


class StudentsPerformance():

  """
  This class implements the students_performance module.
  """

  def __init__(self):
    self.classifier_scores = {}
    self.classifier_pca = {}
    self.classifiers = {}
    self.confusion_matrix_metrics = {}

  """
  Determines optimal number of principal components for each classifier
  and how well it performs using said components
  Positional Arguments: 
  X -- The independent variable(s)
  y -- the response variable
  clfName -- The name of the classifier
  clf -- The classifier
  """
  def ClassifierComparison(self, x, y, clf_name, clf):

    best_pca, best_score = 0,0
    d = x.shape[1]
    for i in range(1,d+1):
      pca = PCA(i).fit(x)
      pca_x = pca.transform(x)
      if cross_val_score(clf, pca_x, y).mean() > best_score:
        best_pca, best_score = i, cross_val_score(clf, pca_x, y).mean()

    #Saving the classifier info, best score, and
    #optimal number of principal components into dictionaries.
    self.classifiers[clf_name] = clf
    self.classifier_scores[clf_name] = round(best_score*100,2)
    self.classifier_pca[clf_name] = best_pca

    print(f"{clf_name} works best with {int(best_pca)} principal components "
        + f"with an average accuracy of {round(best_score*100,2)}")
    return best_score

  """
  Graphs the accuracy of each classifier
  after ClassifierComparison() has been run.
  """
  def GraphComparisons(self):
    if len(self.classifier_scores.keys()) == 0:
      raise Exception(
        "No scores to compare! Try running ClassifierComparison first.")

    plt.bar(range(len(self.classifier_scores)),
           list(self.classifier_scores.values()),
            tick_label = list(self.classifier_scores.keys()))
    plt.xticks(rotation = 45)
    plt.title("How Well Does Each Model Perform?")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.show()

  """
  Creates a confusion matrix for each classifier
  after ClassifierComparison() has been run.
  
  Positional Arguments: 
  X -- The independent variable(s)
  y -- The response variable
  """
  def ConfusionMatrix(self, x, y):

    fig, axs = plt.subplots(2,3, figsize=(20,10))
    rows, cols = [0,0,0,1,1,1], [0,1,2,0,1,2]

    for classifier_name, classifier in self.classifiers.items():

      row, col = rows.pop(0), cols.pop(0)

      #Random state is used to ensure each classifier has the same data.
      #This is useful to ensure the validity of comparisons.
      x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                          test_size=0.33,
                                                          random_state = 15)
      #Fitting each classifier to the data after performing feature reduction
      #using its optimal number of principal components.
      pca = PCA(self.classifier_pca[classifier_name]).fit(x)
      x_train, x_test = pca.transform(x_train), pca.transform(x_test)
      clf = classifier
      clf.fit(x_train, y_train)

      #Getting the predictions from the fitted model
      #and plotting its confusion matrix.
      y_pred = clf.predict(x_test)
      cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
      ConfusionMatrixDisplay(cm, display_labels =
                                    clf.classes_).plot(ax = axs[row][col])

      axs[row][col].set_title(classifier_name, fontsize= 15)
      axs[row][col].set_xlabel("Predicted",fontsize = 15)
      axs[row][col].set_ylabel("True",fontsize = 15)
      axs[row][col].tick_params(axis = "both", labelsize = 15)
      plt.subplots_adjust(hspace = .33)

      #Recording each confusion matrix's metrics.
      tp, fp = cm[1][1], cm[0][1]
      tn, fn = cm[0][0], cm[1][0]
      tpr, fpr = round(tp/(tp+fn)*100, 2), round(fp/(tn+fp)*100,2)
      tnr, fnr = round(tn/(tn+fp)*100, 2), round(fn/(tp+fn)*100,2)
      self.confusion_matrix_metrics[classifier_name] = {
        "TPR": tpr, "FPR": fpr, "TNR": tnr, "FNR": fnr}

    fig.suptitle("Confusion Matrix for Each Classifier", fontsize = 30)
    plt.show()
    return cm

  """
  Reports the matrix of each confusion matrix after confusionMatrix() has been run.
  """
  def ConfusionMatrixMetrics(self):
    print("Confusion Matrix Merics")
    if len(self.confusion_matrix_metrics.keys()) == 0:
      raise Exception(
        "No confusion matrices to work with." +
        "Try running confusionMatrix() first.")
    for name, metrics in self.confusion_matrix_metrics.items():
      for metric, value in metrics.items():
        print (f"{metric} for {name}: {value}")
      print ()

  """
  Uses majority voting to improve results by pooling together response of all the algoirthms
 
  Positional Arguments:
  X -- The independent variable(s)
  y -- The response variable
  """
  def Ensemble(self,x,y):
    #The dictionary will be used to compare classifications.
    y_pred = {}
    best_combo_score = 0

    for r in range(2, len(self.classifiers.keys())+1):
      for combination in itertools.combinations(self.classifiers.keys(),r):
        for classifier in combination:

          #Random state is used to ensure each classifier has the same data.
          #This is useful to ensure the validity of comparisons.
          x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                              test_size=0.25,
                                                              random_state = 33)
          #Fitting each classifier to the data after performing reduction
          #using its optimal number of principal components.
          pca = PCA(self.classifier_pca[classifier]).fit(x)
          x_train, x_test = pca.transform(x_train), pca.transform(x_test)
          clf = self.classifiers[classifier]
          clf.fit(x_train, y_train)

          #Recording the response for comparison.
          y_pred[classifier] = clf.predict(x_test)

        #Final classification after comparing answers.
        final_y_pred = []
        for i in range(len(x_test)):
          passed, failed = [], []

          for classifier_name, predictions in y_pred.items():
            classification = predictions[i]
            if classification == "Passed":
              passed.append(classifier_name)
            else:
              failed.append(classifier_name)

          #If there's a disagreement,
          #weighted voting is used to determine the final classification.
          if len(passed) != 0 and len(failed) != 0:

            weighted_passed, weighted_failed = 0, 0
            for clf_passed in passed:
              weighted_passed += (self.classifier_scores[clf_passed]/100)
            for clf_failed in failed:
              weighted_failed += (self.classifier_scores[clf_failed]/100)

            if weighted_passed >= weighted_failed :
              final_y_pred.append("Passed")
            else:
              final_y_pred.append("Failed")

          else:
            final_y_pred.append(y_pred[classifier_name][i])

        if round(accuracy_score(y_test,final_y_pred)*100,2) > best_combo_score:
          best_combo = combination
          best_combo_score = round(accuracy_score(y_test,final_y_pred)*100,2)
          best_combo_cm = confusion_matrix(y_test, final_y_pred)

    #Creating the final confusion matrix
    ConfusionMatrixDisplay(best_combo_cm,
                           display_labels = ["Failed","Passed"]).plot()
    plt.title("Final Confusion Matrix Using Ensemble Learning", fontsize= 15)
    plt.xlabel("Predicted",fontsize = 15)
    plt.ylabel("True",fontsize = 15)
    plt.tick_params(axis = "both", labelsize = 15)
    plt.show()
    print("The best ensemble is", best_combo)
    print("Final Accuracy:", best_combo_score,"%")
