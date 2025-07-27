import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import scipy.spatial.distance as dst

class BOVW:
    def __init__(self):
        with open('KMeans.pkl', 'rb') as file:
            self.kmeans = pickle.load(file)
        self.bovw_history = []
        self.loopClosure=[]
    def reset(self):
        self.bovw_history = []
    def Histogram(self, desc):
        words = self.kmeans.predict(desc)
        hist, _ = np.histogram(words, bins=np.arange(self.kmeans.n_clusters+1))
        hist = hist / np.linalg.norm(hist)  # Normalize
        return hist
    def historyOfBOVW(self, visual_word, desc, kp):
        self.bovw_history.append([visual_word, desc, kp])

    def LoopChecks(self):
        if len(self.bovw_history)>10:
            for index, [visual_word, desc, kp] in enumerate(self.bovw_history[:-10]):
                similarity = 1 - dst.cosine(visual_word, self.bovw_history[-1][0])
                if similarity > 0.9 and self.GeometricVerification(desc, kp, self.bovw_history[-1][1], self.bovw_history[-1][2]):
                    self.loopClosure.append([index, visual_word, len(self.bovw_history)-1, self.bovw_history[-1][0]])
        return self.loopClosure
    
    def GeometricVerification(self, desc1, kp1, desc2, kp2):
        # initializaing mathes as NORM L2\
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Check if DESC is None
        if desc1 is None or desc2 is None:
            return False
        # Match descreoptors of one and two
        matches = matcher.match(desc1, desc2)
        # if less than 10 matchs then return False
        if len(matches) < 10:
            return False
        src_pts= np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = np.sum(mask)
        # threshold for inliers = 20
        return inliers > 20
        