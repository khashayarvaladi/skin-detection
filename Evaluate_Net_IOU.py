#
###########################################################################################################################################################
import numpy as np
import IOU
import initialization

NumClasses =2
Classes =['skin','non_skin']
Union = np.float64(np.zeros(len(Classes))) #Sum of union
Intersection =  np.float64(np.zeros(len(Classes))) #Sum of Intersection



CIOU,CU=IOU.GetFMeasure(initialization.mask,initialization.groundTruthMask.squeeze(),len(Classes),Classes) #Calculate intersection over union
Intersection+=CIOU*CU
Union+=CU

#-----------------------------------------Print results--------------------------------------------------------------------------------------
print("---------------------------Mean Prediction----------------------------------------")
print("---------------------IOU=Intersection Over Inion----------------------------------")
for i in range(len(Classes)):
        if Union[i]>0: print(Classes[i]+"\t"+str(Intersection[i]/Union[i]))


##################################################################################################################################################

print("Finished")