# Author: Juho Kemppainen
# 4/2022
# K-means, by default pre-set 2 clusters, amount of iterations can be changed
# different colors of data-points on plot describe init, moving and final position of clusters
# output is "k-means result.txt" where you can match clusters (4. column) to ground truth (3. column)
# ZeroDivisionError: division by zero - error may occur sometimes. Need to rerun the program

import numpy as np                      # pip3 install numpy
import pandas as pd                     # pip3 install pandas
import matplotlib.pyplot as plt         # pip3 install matplotlib
import math
import random

fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(12, 6))     # 3,3 - amount of subplots

file = "jain.txt"       # data file
x_points = []
y_points = []
label = []
newCentroid = []
dictionary = {'x':x_points, 'y': y_points,'label': label, 'newCentroid': newCentroid}   # dictionary (key:value) with 4 columns

###############################################   text file process   #####################################################################
def data_processing(file):

    for line in open(file):
        a = float(line.split()[0])  # picking up 1. column and converting string -> float. CONVERTING IS IMPORTANT FOR PLOT
        b = float(line.split()[1])
        c = line.split()[2]

        x_points.append(a)              # x
        y_points.append(b)              # y
        label.append(int(c))            # ground true 
        newCentroid.append(int(c))      # new centroids, copy of ground true <- need for success work of dataframe
        #print(a)


data_processing(file)
df = pd.DataFrame(dictionary)
#print(df)                              # checks dataframe
#print(df.to_string())  

c1 = [random.uniform(0.1, 45), random.uniform(0.1, 30)]     # 2 centroids RANDOMIZED!!!
c2 = [random.uniform(0.1, 45), random.uniform(0.1, 30)]     

###############################################   k-means   ##################################################################################
def k_means(c1, c2, iterations):
    axs1.scatter(x_points, y_points)
    axs1.title.set_text("k-means")
    global newCentroid
    #newCentroid = []
    distance = 0
    temp_1 = 0
    temp_2 = 0
    
    axs1.scatter(c1[0], c1[1], color='red', label='Centroid random init')      # plotting centroids
    axs1.scatter(c2[0], c2[1], color='red')
    axs1.text(c1[0], c1[1], 'C1', fontsize=12)
    axs1.text(c2[0], c2[1], 'C2', fontsize=12)
    
    print('\nK-MEANS\n')
    
    for x in range(iterations):
        print('\nCentroid points before ', x+1,'iteration')
        print('C1: ', round(c1[0], 2), round(c1[1], 2))
        print('C2: ', round(c2[0], 2), round(c2[1], 2), '\n')
        newCentroid = []

        X_new_Centroid_1_Values = []            # centroid 1 x-value list
        Y_new_Centroid_1_Values = []            # centroid 1 y-value list
        
        X_new_Centroid_2_Values = []
        Y_new_Centroid_2_Values = []
    
    ##################################  measuring distances from data-point to clusters and pick minimum ################################
        for i in range(0, len(x_points)):

            a = x_points[i], y_points[i]    # data-point
            
            temp_1 = math.dist(c1,a)        # measuring distances between centroids and data-point
            temp_2 = math.dist(c2,a)
            distance = min(temp_1,temp_2)   # taking min distance of data-point to c1 & c2
            
            if distance == temp_1:
                #print('C1', temp_1)
                newCentroid.insert(i, 1)
            
            elif distance == temp_2 :
                #print('C2', temp_2)
                newCentroid.insert(i, 2)
        
        #print(newCentroid)                  # prints each data-point's centroid
        #print(df)   
        
    ################################### calculating new coordinates to centroids ##################################################
        for i, j in df.iterrows():      # looping dataframe, i = row number, j = column values
           
            if newCentroid[i] == 1:               # if newCentroid[i] matches centroid number 1
                X_new_Centroid_1_Values.append(j[0])
                Y_new_Centroid_1_Values.append(j[1])
                #print('CENTROID 1:', i,j)
                
                
            elif newCentroid[i] == 2:               # if newCentroid[i] matches centroid number 2
                X_new_Centroid_2_Values.append(j[0])
                Y_new_Centroid_2_Values.append(j[1])
                #print('CENTROID 2:', i,j)


        c1 = sum(X_new_Centroid_1_Values) / len(X_new_Centroid_1_Values), sum(Y_new_Centroid_1_Values) / len(Y_new_Centroid_1_Values)
        c2 = sum(X_new_Centroid_2_Values) / len(X_new_Centroid_2_Values), sum(Y_new_Centroid_2_Values) / len(Y_new_Centroid_2_Values)
        
        axs1.scatter(c1[0], c1[1], color='black')      # plotting moving centroids
        axs1.scatter(c2[0], c2[1], color='black')
        
    print('\nFinal centroid points')
    print('C1: ', round(c1[0], 2), round(c1[1], 2))
    print('C2: ', round(c2[0], 2), round(c2[1], 2), '\n')    
    axs1.scatter(c1[0], c1[1], color='orange', label='Centroid final')      # plotting final centroids
    axs1.scatter(c2[0], c2[1], color='orange')
    axs1.text(c1[0], c1[1], 'C1', fontsize=12)
    axs1.text(c2[0], c2[1], 'C2', fontsize=12)
    axs1.legend()


#plt.show()

resultFile = 'k-means result.txt'               
def createTxtFile(file):                   # generates "k-means result.txt" file with results
    a = np.array(x_points)
    b = np.array(y_points)
    c = np.array(label)
    d = np.array(newCentroid)
    data = np.column_stack([a,b,c,d])
    #print(newCentroid)                     # # prints each data-point's centroid
    #print(data)                             # print results
    np.savetxt(file, data, fmt=['%1.2f\t', '%1.2f\t', '%d\t', '%d'])        # 1.2f - float with two decimals, \n - tab space

#createTxtFile(resultFile)

###############################################  fuzzy c-means   ##################################################################################
def fuzzy_C_means():
    file = "butterfly.txt"
    x_points = []               # 5 lists for dictionary
    y_points = []
    nearest_C = []
    list1 = []
    list2 = []
    
    X_1_Values = []             # centroid 1 x-value list
    Y_1_Values = []             # centroid 1 y-value list
    X_2_Values = []
    Y_2_Values = []
    
    c1 = [random.uniform(0, 6), random.uniform(0, 5)]     # 2 centroids RANDOMIZED!!!
    c2 = [random.uniform(0, 6), random.uniform(0, 5)]
    c1_init = c1
    c2_init = c2 
    
    iterating = False
    print('\nFUZZY C MEANS\n')
    
    def reading_data():

        axs2.scatter(c1[0], c1[1], color='red', label='Centroid random init')      # plotting centroids
        axs2.scatter(c2[0], c2[1], color='red')
        axs2.text(c1[0], c1[1], 'C1', fontsize=12)
        axs2.text(c2[0], c2[1], 'C2', fontsize=12)  
        
        for line in open(file):
            a = float(line.split()[0])  # picking up 1. column and converting string -> float. CONVERTING IS IMPORTANT FOR PLOT
            b = float(line.split()[1])

            x_points.append(a)              # x
            y_points.append(b)              # y
            #list1.append(0)
            #list2.append(0)
            #nearest_C.append(0)

    reading_data()
    axs2.scatter(x_points, y_points)
    axs2.title.set_text("Fuzzy c-means")
    
    ################################## step 1. -  measuring distances from data-point to clusters and pick minimum ################################
    def measure(c1, c2):


        for i in range(0, len(x_points)):

            a = x_points[i], y_points[i]    # data-point
            
            temp_1 = math.dist(c1,a)        # measuring distances between centroids and data-point
            temp_2 = math.dist(c2,a)
            distance = min(temp_1,temp_2)   # taking min distance of data-point to c1 & c2
            
            if distance == temp_1:
                #print('C1', temp_1)
                nearest_C.insert(i, 1)
                list1.insert(i, 1)          # filling other columns
                list2.insert(i, 0)
                
            
            elif distance == temp_2 :
                #print('C2', temp_2)
                nearest_C.insert(i, 2)      
                list1.insert(i, 0)          # filling other columns
                list2.insert(i, 1)
                
        for i in range(0, len(x_points)):      
        
            if nearest_C[i] == 1:               # if newCentroid[i] matches centroid number 1
                X_1_Values.append(list1[0])
                Y_1_Values.append(list2[1])
                #print('CENTROID 1:', i,j)
                
                
            elif nearest_C[i] == 2:               # if newCentroid[i] matches centroid number 2
                X_2_Values.append(list1[0])
                Y_2_Values.append(list2[1])
                #print('CENTROID 2:', i,j)
        
        c1 = sum(X_1_Values) / len(X_1_Values), sum(Y_1_Values) / len(Y_1_Values)
        c2 = sum(X_2_Values) / len(X_2_Values), sum(Y_2_Values) / len(Y_2_Values)
        print(c1,c2)
        
        axs2.scatter(c1[0], c1[1], color='green')      # plotting moving centroids
        axs2.scatter(c2[0], c2[1], color='green')

    
    measure(c1, c2)
    
    d = {'x-axel':x_points, 'y-axel':y_points, 'Centroid': nearest_C, 'c1':list1, 'c2':list2}
    ddf = pd.DataFrame(d)
    print(ddf)

    # while iterating:
    
    ################################### step 2. - computing new coordinates centroids (u * data-point)  ############################################
    def centroids():
        print('yo')
        
        #TODO   


    
    centroids()

    ####################################### step 3 - computing membership degress, C=2   ################################################################
    def step_3_membership_metrics():
        for i in range(0, len(x_points)):
            
            if nearest_C[i] == 1:
                a = x_points[i], y_points[i]    # data-point
                
                temp_1 = math.dist(c1,a)        # measuring distances between centroids and data-point
                temp_2 = math.dist(c2,a)
                
                degree = round((pow(pow((temp_1/temp_1), 2) + pow((temp_1/temp_2), 2), -1)), 2)
                anti_degree = round(1-degree, 2)
                list1[i] = degree               # updating U-matrix
                list2[i] = anti_degree
                #print('degree', degree, anti_degree)
                
            elif nearest_C[i] == 2:
                a = x_points[i], y_points[i]    # data-point
                
                temp_1 = math.dist(c1,a)        # measuring distances between centroids and data-point
                temp_2 = math.dist(c2,a)
                
                degree = round((pow(pow((temp_1/temp_1), 2) + pow((temp_1/temp_2), 2), -1)), 2)
                anti_degree = round(1-degree, 2)
                list1[i] = degree               # updating U-matrix
                list2[i] = anti_degree
                #print('degree', degree, anti_degree)
                
        d = {'x-axel':x_points, 'y-axel':y_points, 'Centroid': nearest_C, 'c1':list1, 'c2':list2}   # updating dictionary and dataframe
        ddf = pd.DataFrame(d)
        print(ddf)

    step_3_membership_metrics()
    
    ####################################### step 4 - not working   ################################################################
    dic = {'list1': list1, 'list2': list2}
    pddd = pd.DataFrame(dic)
    res = list1.copy()
    
    def convergence():
        #print(pddd)

        for i, j in pddd.iterrows():
            temp = 1 - (pow(j[0], 2) + pow(j[1], 2))
            temp1 = round(temp, 5)
            res[i] = temp1
            maximum = max(res)
          
        if maximum < 0.01:      
            False               # True or False will define iterations of FCM
            #print(maximum)
        else:
            True
            #print(maximum)         # don't undestand which is max and epsilon

    convergence()
    
    
iterations = 5                            # k-means iterations
k_means(c1, c2, iterations)
    
fuzzy_C_means()
plt.show()




