import numpy as np
import BagLearner , LinRegLearner
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.bags = [BagLearner.BagLearner(learner=LinRegLearner.LinRegLearner,kwargs={}) for i in range(0,20)]
    def author(self):
        return 'snagamalla3' # replace tb34 with your Georgia Tech username
    def addEvidence(self,dataX,dataY):
        [bag.addEvidence(dataX,dataY) for bag in self.bags]
    def query(self,points):
        ans= []
        [ans.append(bag.query(points)) for bag in self.bags]
        return np.mean(ans,axis=0)
if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
