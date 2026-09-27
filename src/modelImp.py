import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#------------------------------------------absolute path----------------


import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

data=pd.read_csv(os.path.join(DATA_DIR, 'dataAfterPre.csv'))


#------------------------------Data Preperaion-----------------------------
def dataPrep(data):
    data=np.array(data)
    data=(data-(np.mean(data,axis=0)))/(np.std(data,axis=0))
    y=np.array(data[:,0]).reshape(-1,1)
    x=np.array(data[:,1:])
    m=y.shape[0]
    n=x.shape[1]
    
    x_bias=np.ones((m,1))
    X_matrix=np.hstack((x_bias,x))
    
    return y,X_matrix


#------------------------------model-----------------------------


y,X_matrix=dataPrep(data)

def fun(X_matrix,y,alpha,grad_check,cost_check,max_epochs):
    thetas=[]
    costs=[]
    epcoh_costs=[]
    b=32
    n=X_matrix.shape[1]
    m=X_matrix.shape[0]
    theta_vector=np.zeros((n,1))  #initialization   1

    number_of_mini_batchs=int(m/32)
    number_of_mini_batchs=int(m/b)
    for i in range(max_epochs):
        
        current_b=b
        for l in range(number_of_mini_batchs):
            thetas.append(theta_vector)
            
            start=l*b
            end=(l+1)*b
            batch_X_matrix=X_matrix[start:end,:]
            batch_y=y[start:end,]
            
           
            if  l+1==number_of_mini_batchs:
                batch_X_matrix=X_matrix[start:,:]
                batch_y=y[start:,]
                current_b=m-(l*b)

            else :
                batch_X_matrix=X_matrix[start:end,:]
                batch_y=y[start:end,]
                current_b=b

            
            y_hat=(batch_X_matrix)@(theta_vector) #model    2
            
            error=y_hat-batch_y
            cost_fun=((np.linalg.norm(error))**2)/(2*current_b) #cost function    3
            costs.append(cost_fun)
        
            grad_vector=((batch_X_matrix.T)@error)/current_b    #grad 4
        
            theta_vector=theta_vector - (alpha*grad_vector) #update 5
        last_thetas=theta_vector
        epcoh_costs.append(cost_fun)
        if (np.linalg.norm(grad_vector)<grad_check):
            break
        elif (i>0 and np.absolute(epcoh_costs[i]-epcoh_costs[i-1])<cost_check):
            break
    return thetas,costs,last_thetas,i







if __name__ == "__main__":
    thetas,costs,last_thetas,i=fun(X_matrix,y,0.001,0.001,0.001,500)
    print(i)
    # print(costs[-1])
    plt.plot(costs)
    plt.show()

    y_predict=X_matrix@last_thetas
    r2=r2_score(y,y_predict)
    print(r2)




    
