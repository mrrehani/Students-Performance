from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class StudentsPerformance():

    def __init__(self):
        self.classifierScores = {}
        self.classifierPCA = {}
        self.classifiers = {}
        self.confusionMatrixMetrics = {}
    
    def ClassifierComparison(self, X,y,clfName, clf):
        bestPCA, bestScore = 0,0
        k, d = X.shape
        for i in range(1,d+1):
            pca = PCA(i).fit(X)
            pcaX = pca.transform(X)
            if cross_val_score(clf, pcaX, y).mean() > bestScore:
                bestPCA, bestScore = i, cross_val_score(clf, pcaX, y).mean()
        self.classifiers[clfName] = clf
        self.classifierScores[clfName] = round(bestScore*100,2)
        self.classifierPCA[clfName] = bestPCA
        print("%s works best with %s principal components with an average accuracy of %s" %(clfName, int(bestPCA), round(bestScore*100,2)))
        return bestScore
       
    def graphComparisons(self):
        if len(self.classifierScores.keys()) == 0:
            raise Exception("No scores to compare!")

        plt.bar(range(len(self.classifierScores)), list(self.classifierScores.values()), tick_label = list(self.classifierScores.keys()))
        plt.xticks(rotation = 45)
        plt.title("How Well Does Each Model Perform?")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.show()
        
    def confusionMatrix(self, X, y):
        fig, axs = plt.subplots(2,3, figsize=(20,10)) 
        rows, cols = [0,0,0,1,1,1], [0,1,2,0,1,2]
        

        for classifier in self.classifiers:
            row, col = rows.pop(0), cols.pop(0)
            pca = PCA(self.classifierPCA[classifier]).fit(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 15)
            X_train, X_test = pca.transform(X_train), pca.transform(X_test)

            clf = self.classifiers[classifier]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            CM = confusion_matrix(y_test, y_pred, labels = clf.classes_)
            TP, FP = CM[1][1], CM[0][1]
            TN, FN = CM[0][0], CM[1][0]
            TPR, FPR = round(TP/(TP+FN)*100, 2), round(FP/(TN+FP)*100,2)
            TNR, FNR = round(TN/(TN+FP)*100, 2), round(FN/(TP+FN)*100,2)
            self.confusionMatrixMetrics[classifier] = {"TPR": TPR, "FPR": FPR, "TNR": TNR, "FNR": FNR}

            ConfusionMatrixDisplay(CM, display_labels = clf.classes_).plot(ax = axs[row][col])
            axs[row][col].set_title(classifier, fontsize= 15)
            axs[row][col].set_xlabel("Predicted",fontsize = 15)
            axs[row][col].set_ylabel("True",fontsize = 15)
            axs[row][col].tick_params(axis = "both", labelsize = 15)
            plt.subplots_adjust(hspace = .33)
        fig.suptitle("Confusion Matrix for Each Classifier", fontsize = 30)
        plt.show()
        return CM
    
    def ensemble(self,X,y):
        y_pred = {}
        for classifier in self.classifiers:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 33)
            pca = PCA(self.classifierPCA[classifier]).fit(X)
            X_train, X_test = pca.transform(X_train), pca.transform(X_test)
            clf = self.classifiers[classifier]
            clf.fit(X_train, y_train)
            y_pred[classifier] = clf.predict(X_test)

        final_y_pred = []
        for i in range(len(X_test)):
            passed, failed = [], []
            for clf in y_pred:
                classification = y_pred[clf][i]
                if classification == "Passed":
                    passed.append(clf)
                else:
                    failed.append(clf) 
            if len(passed) != 0 and len(failed) != 0:
                if len(failed) >= 3:
                    final_y_pred.append("Failed")
                else:
                    final_y_pred.append("Passed")
            else:
                final_y_pred.append(y_pred[clf][i])

        CM = confusion_matrix(y_test, final_y_pred)
        ConfusionMatrixDisplay(CM).plot()
        plt.title("Final Confusion Matrix Using Ensemble Learning", fontsize= 15)
        plt.xlabel("Predicted",fontsize = 15)
        plt.ylabel("True",fontsize = 15)
        plt.tick_params(axis = "both", labelsize = 15)
        plt.show()
        print("Final Accuracy:", round(accuracy_score(y_test,final_y_pred)*100,2),"%")

                
