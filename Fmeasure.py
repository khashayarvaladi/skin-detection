import numpy as np
import scipy.misc as misc

def GetFMeasure(Pred,GT,NumClasses,ClassNames=[], DisplyResults=False): #Given A ground true and predicted labels return the intersection over union for each class
    # and the union for each class
    ClassF1=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        TP = np.float32(np.sum((Pred==GT)*(GT==i)))
        FP = np.float32(np.sum((Pred!=GT)*(GT==i)))
        FN = np.float32(np.sum((Pred!=GT)*(GT!=i)))

        Intersection = 2 * TP   # Calculate intersection
        Union=  2*TP + FP+FN# Calculate Union
        if Union>0:
            ClassF1[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union

    #------------Display results-------------------------------------------------------------------------------------
    if DisplyResults:
       for i in range(len(ClassNames)):
            print(ClassNames[i]+") "+str(ClassF1[i]))
       print("Mean Classes F1) "+str(np.mean(ClassF1)))
       print("Image Predicition Accuracy)" + str(np.float32(np.sum(Pred == GT)) / GT.size))
    #-------------------------------------------------------------------------------------------------

    return ClassF1, ClassWeight





