from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import itertools



class StudentsPerformance():

    def __init__(self):
        self.classifierScores = {}
        self.classifierPCA = {}
        self.classifiers = {}
        self.confusionMatrixMetrics = {}
    
    """
    Determines optimal number of principal components for each classifier and how well it performs using said components

    Positional Arguments: 
    X -- The independent variable(s)
    y -- the response variable
    clfName -- The name of the classifier
    clf -- The classifier
    """
    def ClassifierComparison(self, X, y, clfName, clf):
        bestPCA, bestScore = 0,0
        k, d = X.shape
        for i in range(1,d+1):
            pca = PCA(i).fit(X)
            pcaX = pca.transform(X)
            if cross_val_score(clf, pcaX, y).mean() > bestScore:
                bestPCA, bestScore = i, cross_val_score(clf, pcaX, y).mean()
        #Saving the classifier info, best score, and optimal number of principal components into dictionaries. 
        self.classifiers[clfName] = clf
        self.classifierScores[clfName] = round(bestScore*100,2)
        self.classifierPCA[clfName] = bestPCA
        print("%s works best with %s principal components with an average accuracy of %s" %(clfName, int(bestPCA), round(bestScore*100,2)))
        return bestScore
    
    """
    Graphs the accuracy of each classifier after ClassifierComparison() has been run.
    """
    def graphComparisons(self):
        if len(self.classifierScores.keys()) == 0:
            raise Exception("No scores to compare! Try running ClassifierComparison first.")

        plt.bar(range(len(self.classifierScores)), list(self.classifierScores.values()), tick_label = list(self.classifierScores.keys()))
        plt.xticks(rotation = 45)
        plt.title("How Well Does Each Model Perform?")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.show()
        
    """
    Creates a confusion matrix for each classifier after ClassifierComparison() has been run.
    
    Positional Arguments: 
    X -- The independent variable(s)
    y -- The response variable
    """    
    def confusionMatrix(self, X, y):

        fig, axs = plt.subplots(2,3, figsize=(20,10)) 
        rows, cols = [0,0,0,1,1,1], [0,1,2,0,1,2]

        for classifier in self.classifiers:

            row, col = rows.pop(0), cols.pop(0)
            
            #Random state is used to ensure each classifier has the same data. This is useful to ensure the validity of comparisons.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 15)
            #Fitting each classifier to the data after performing feature reduction using its optimal number of principal components.
            pca = PCA(self.classifierPCA[classifier]).fit(X) 
            X_train, X_test = pca.transform(X_train), pca.transform(X_test)
            clf = self.classifiers[classifier]
            clf.fit(X_train, y_train)
            
            #Getting the predictions from the fitted model and plotting its confusion matrix.
            y_pred = clf.predict(X_test)
            CM = confusion_matrix(y_test, y_pred, labels = clf.classes_)
            ConfusionMatrixDisplay(CM, display_labels = clf.classes_).plot(ax = axs[row][col])
            axs[row][col].set_title(classifier, fontsize= 15)
            axs[row][col].set_xlabel("Predicted",fontsize = 15)
            axs[row][col].set_ylabel("True",fontsize = 15)
            axs[row][col].tick_params(axis = "both", labelsize = 15)
            plt.subplots_adjust(hspace = .33)

            #Recording each confusion matrix's metrics.
            TP, FP = CM[1][1], CM[0][1]
            TN, FN = CM[0][0], CM[1][0]
            TPR, FPR = round(TP/(TP+FN)*100, 2), round(FP/(TN+FP)*100,2)
            TNR, FNR = round(TN/(TN+FP)*100, 2), round(FN/(TP+FN)*100,2)
            self.confusionMatrixMetrics[classifier] = {"TPR": TPR, "FPR": FPR, "TNR": TNR, "FNR": FNR}

        fig.suptitle("Confusion Matrix for Each Classifier", fontsize = 30)
        plt.show()
        return CM
    
    """
    Reports the matrix of each confusion matrix after confusionMatrix() has been run.
    """
    def confusionMatrix_metrics(self):
        print("Confusion Matrix Merics")
        if len(self.confusionMatrixMetrics.keys()) == 0:
            raise Exception("No confusion matrices to work with. Try running confusionMatrix() first.")
        for classifier in self.confusionMatrixMetrics:
            for metric in self.confusionMatrixMetrics[classifier]:
                print ("%s for %s: %s" %(metric, classifier, self.confusionMatrixMetrics[classifier][metric]))
            print ()

    """
    Uses majority voting to improve results by pooling together response of all the algoirthms
    
    Positional Arguments: 
    X -- The independent variable(s)
    y -- The response variable
    """
    def ensemble(self,X,y):
        #The dictionary will be used to compare classifications.
        y_pred = {}
        bestComboScore = 0

        for r in range(2, len(self.classifiers.keys())+1):
            for combination in itertools.combinations(self.classifiers.keys(),r):
                for classifier in combination:
                
                    #Random state is used to ensure each classifier has the same data. This is useful to ensure the validity of comparisons.
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 33)
                    #Fitting each classifier to the data after performing feature reduction using its optimal number of principal components.
                    pca = PCA(self.classifierPCA[classifier]).fit(X)
                    X_train, X_test = pca.transform(X_train), pca.transform(X_test)
                    clf = self.classifiers[classifier]
                    clf.fit(X_train, y_train)
                    
                    #Recording the response for comparison.
                    y_pred[classifier] = clf.predict(X_test)

                #Final classification after comparing answers.
                final_y_pred = []
                for i in range(len(X_test)):
                    passed, failed = [], []

                    for clf in y_pred:
                        classification = y_pred[clf][i]
                        if classification == "Passed":
                            passed.append(clf)
                        else:
                            failed.append(clf) 

                    #If there's a disagreement, majority voting is used to determine the final classification.
                    if len(passed) != 0 and len(failed) != 0:
                        
                        weightedPassed, weightedFailed = 0, 0
                        for clfPassed in passed:
                            weightedPassed += (self.classifierScores[clfPassed]/100)
                        for clfFailed in failed:
                            weightedFailed += (self.classifierScores[clfFailed]/100)

                        if weightedPassed >= weightedFailed :
                            final_y_pred.append("Passed")
                        else:
                            final_y_pred.append("Failed")

                    else:
                        final_y_pred.append(y_pred[clf][i])

                if round(accuracy_score(y_test,final_y_pred)*100,2) > bestComboScore:
                    bestCombo, bestComboScore = combination, round(accuracy_score(y_test,final_y_pred)*100,2)
                    bestComboCM = confusion_matrix(y_test, final_y_pred)
        
        #Creating the final confusion matrix
        ConfusionMatrixDisplay(bestComboCM, display_labels = ["Failed","Passed"]).plot()
        plt.title("Final Confusion Matrix Using Ensemble Learning", fontsize= 15)
        plt.xlabel("Predicted",fontsize = 15)
        plt.ylabel("True",fontsize = 15)
        plt.tick_params(axis = "both", labelsize = 15)
        plt.show()
        print("The best ensemble is", combination)
        print("Final Accuracy:", bestComboScore,"%")
            

                        
