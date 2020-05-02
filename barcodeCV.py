import numpy as np
import argparse
import cv2

from matplotlib import pyplot as plt


negEdgesSums = []
posEdgesSums = []

threshValue = 135


edgeMinValue = 100
edgeMaxValue = 200

apetureSize = 10

for i in range(10):

    posFolder = "/Users/douglasjohnson/Desktop/RoboCV/Positives/"
    negFolder = "/Users/douglasjohnson/Desktop/RoboCV/Negatives/"
    posPath = posFolder + "barcode_0{}.jpg".format(i)
    negPath = negFolder + "negative_0{}.jpg".format(i)




    posImg = cv2.resize(cv2.imread(posPath), (500, 500))
    negImg = cv2.resize(cv2.imread(negPath), (500, 500))

    #print("Image Size:", posImg.shape)
    #print("Image Size:", negImg.shape)
    posGray = cv2.cvtColor(posImg, cv2.COLOR_BGR2GRAY)
    negGray = cv2.cvtColor(negImg, cv2.COLOR_BGR2GRAY)

#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    posRet, posThresh = cv2.threshold(posGray,threshValue,255, cv2.THRESH_BINARY)
    negRet, negThresh = cv2.threshold(negGray,threshValue,255, cv2.THRESH_BINARY)

    posEdges = cv2.Canny(posThresh, edgeMinValue, edgeMaxValue)
    negEdges = cv2.Canny(negThresh, edgeMinValue, edgeMaxValue)

    cv2.imshow("posEdges", posEdges)
    cv2.imshow("negEdges", negEdges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    negEdgesSums.append(np.sum(negEdges)//255)
    posEdgesSums.append(np.sum(posEdges)//255)



print("")
print("STATS\n")

print("NEGATIVES:\n")
for i in range(10):
    print("Negative 0{} Edge Sum:   ".format(i), int(negEdgesSums[i]))

print("\nPOSITIVES:\n")
for i in range(10):
    print("Positive 0{} Edge Sum:   ".format(i), int(posEdgesSums[i]))

print("")

print("SUMS:\n")
print("Total Sum of Negative Edges:   ", "{:.2e}".format(int(sum(negEdgesSums))))
print("Total Sum of Positive Edges:   ", "{:.2e}".format(int(sum(posEdgesSums))))

print("Total Negative Edges/Total Positive Edges:   ", round(sum(negEdgesSums)/sum(posEdgesSums), 2), "\n")