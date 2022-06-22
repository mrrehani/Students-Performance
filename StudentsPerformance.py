from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class StudentsPerformance():

    def __init__(self):
        self.classifierScores = {}
        self.classifierPCA = {}
        self.classifiers = {}
    
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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

            clf = self.classifiers[classifier]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            CM = confusion_matrix(y_test, y_pred, labels = clf.classes_)
            ConfusionMatrixDisplay(CM, display_labels = clf.classes_).plot(ax = axs[row][col])
            axs[row][col].set_title(classifier, fontsize= 15)
            axs[row][col].set_xlabel("Predicted",fontsize = 15)
            axs[row][col].set_ylabel("True",fontsize = 15)
            axs[row][col].tick_params(axis = "both", labelsize = 15)
            plt.subplots_adjust(hspace = .33)
        fig.suptitle("Confusion Matrix for Each Classifier", fontsize = 30)
        plt.show()
                
