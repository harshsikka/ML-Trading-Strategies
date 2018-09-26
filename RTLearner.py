import numpy as np

class RTLearner(object):

    dt_table = {}



    def __init__(self, leaf_size,verbose = False):
        self.counter = 0
        self.leaf_size = leaf_size
        pass # move along, these aren't the drones you're looking for

    def author(self):
        return 'snagamalla3' # replace tb34 with your Georgia Tech username

    def best_feature(self,dataX,dataY):
        factors = dataX.shape[1]
        return  np.random.randint(factors, size=1)[0]


    def build_table(self,dataX,dataY):
        if dataX.shape[0] <= self.leaf_size or len(np.unique(dataY)) == 1:
            elem = np.mean(dataY)
            return np.array([[-1 , elem , None , None]])
        factor = self.best_feature(dataX,dataY)
        splitVal = np.median(dataX[:,factor])

        left_data = dataX[:,factor] <= splitVal
        right_data = np.logical_not(left_data)

        if np.all(left_data) or np.all(right_data):
            elem = np.mean(dataY)
            return np.array([[-1 , elem , None , None]])

        left_x_val , left_y_val =dataX[left_data],dataY[left_data]
        right_x_val , right_y_val=dataX[right_data],dataY[right_data]
        left_tree = self.build_table(left_x_val,left_y_val)
        right_tree = self.build_table(right_x_val,right_y_val)

        root = np.array([[factor,splitVal,1,left_tree.shape[0]+1]])

        left_joined = np.append(root,left_tree,axis=0)
        right_joined = np.append(left_joined,right_tree,axis=0)

        return right_joined


    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # slap on 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
        newdataX[:,0:dataX.shape[1]]=dataX

        # build and save the model

        self.dtt = self.build_table(dataX,dataY)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        ans = []
        for point in points:
            node = self.dtt[0]
            index = 0
            while(node[0] != -1):
                condition_factor = int(node[0])
                split_val = node[1]
                left = int(node[2])
                right = int(node[3])

                if point[condition_factor] <= split_val:
                    index = index + left
                    node = self.dtt[index]
                else:
                    index =index + right
                    node = self.dtt[index]

            ans.append(node[1])
        return ans

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
