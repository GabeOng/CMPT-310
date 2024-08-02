# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# Mira implementation
import util
PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            cGrid = [0.001, 0.002, 0.004, 0.008]
        else:
            cGrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, cGrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, cGrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        bestAccuracyCount = -1  # best accuracy so far on validation set
        cGrid.sort(reverse=True)
        bestParams = cGrid[0]
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        for c in bestParams:
            self.initializeWeightsToZero()
            for j in range(self.max_iterations):
                for i in range(len(trainingData)):
                    pred=util.Counter()
                    for l in self.legalLabels:
                        pred[l]=trainingData[i]*self.weights[l]
                    py=pred.argMax()
                    if py !=trainingLabels[i]:
                        num=((self.weights[py]-self.weights[trainingLabels[i]])*trainingData[i]+1)
                        tao = min(c, 0.5*num/(trainingData[i]*trainingData[i]))
                        step = trainingData[i].copy()
                        step.divideAll(1.0/tao)
                        self.weights[py]-=step
                        self.weights[trainingLabels[i]]+=step
        cVal=sum([1 for y1, y2 in zip(self.classify(validationData), validationLabels) if y1 == y2])
        if cVal > bestAccuracyCount:
            bestParams, bestAccuracyCount = self.weights.copy(), cVal
        return bestParams
        print("finished training. Best cGrid param = ", bestParams)

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        "*** YOUR CODE HERE ***"
        for d in data:
            v = util.Counter()
            for l in self.legalLabels:
                v[l] = self.weights[l] * d
            guesses.append(v.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"

        return self.weights[label].sortedKeys()[:100]
