# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 07:29:20 2021

@author: bahramis
"""


import PriceFunc
import UsefulFunc
import matplotlib.pyplot as plt
import NNmodelPolicy as nnPolicy
import NNmodelValue as nnValue
import torch
from torch import optim
import numpy as np
from scipy.stats import truncnorm
from numpy.random import seed
from numpy.random import randint

Appnum=1
PowerSlice=0.1
NumSlice=int(1+(1/PowerSlice))
bet=0.8
Day=1
D=96
NumDays=60
T=D*NumDays
HomeNum=1
Price=np.zeros(T)
Price=np.tile(PriceFunc.PriceFunc(D),int(T/D))
disCoef=30
billCoef=1
Flag = np.zeros(Appnum)

Tmin = np.zeros((Appnum,T))
Tmax = np.zeros((Appnum,T))

Tairmin = 10*np.ones((Appnum,T))
Tairmax = 30*np.ones((Appnum,T))

Tsetmin = np.zeros((Appnum,T))
Tsetmax = 30*np.ones((Appnum,T))

w = 4*np.ones((Appnum,T))  #Unit:  cents/C
test=1
z=np.exp(-300/130)

#%%
from datetime import datetime, timedelta
timearray=np.zeros((Appnum,T))
date_N_days_ago = datetime.now() - timedelta(minutes=10*T)
t = np.arange(date_N_days_ago, datetime.now(), timedelta(minutes=10)).astype(datetime)

#%%
N=np.zeros((Appnum,T))
for i in range(0,Appnum):
    for t in range(0, T-1):
        if t%96<=48:
            if t%3!=0:
                N[i,t]=N[i,t-1]
            else:
                N[i,t]= randint(0, 4)
                while t>=0 and np.absolute(N[i,t]-N[i,t-1])>1:
                    N[i,t]= randint(0, 4)
        else:
            N[i,t]=0
k=4
major_ticks = np.arange(0, (k+1)*96, 96)
plt.plot(N[0,0:96*k], 'bo-', linewidth=2, markersize=3)
plt.xticks(major_ticks)
plt.ylabel('Occupancy')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 50))
plt.show()

#%%
Tout = np.zeros((Appnum,T))
for i in range(0,Appnum):
    for t in range(0, T-1):
        if t%96<=48:
            if t%3!=0:
                Tout[i,t]=Tout[i,t-1]
            else:
                Tout[i,t]=  0.25*randint(56, 64)
        else:
            if t%3!=0:
                Tout[i,t]=Tout[i,t-1]
            else:
                Tout[i,t]=  0.25*randint(44, 52)



k=4
major_ticks = np.arange(0, (k+1)*96, 96)
plt.plot(Tout[0,0:96*k], 'go-', linewidth=2, markersize=3)
plt.xticks(major_ticks)
plt.ylabel('Outdoor Temperature')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 50))
plt.show()

#%%
Tdes = np.zeros((Appnum,T))
for i in range(0,Appnum):
    for t in range(0, T-1):
        if t%96<=48:
            if t%3!=0:
                Tdes[i,t]=Tdes[i,t-1]
            else:
                Tdes[i,t]= randint(24, 25) 
        else:
            if t%3!=0:
                Tdes[i,t]=Tdes[i,t-1]
            else:
                Tdes[i,t]=  0

k=4
major_ticks = np.arange(0, (k+1)*96, 96)
plt.plot(Tdes[0,0:96*k], 'ro-', linewidth=2, markersize=3)
plt.xticks(major_ticks)
plt.ylabel('Desirable Temperature')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 50))
plt.show()

#%%
Tset_manual = np.zeros((Appnum,T))
for i in range(0,Appnum):
    for t in range(0, T-1):
        if t%96<=48:
                Tset_manual[i,t]= 24 
        else:
                Tset_manual[i,t]= 5 
k=4
major_ticks = np.arange(0, (k+1)*96, 96)
plt.plot(Tset_manual[0,0:96*k], 'yo-', linewidth=2, markersize=3)
plt.xticks(major_ticks)
plt.ylabel('Manual Setpoint Temperature')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 50))
plt.show()

#%%
Tin_manual=np.zeros((Appnum,T))
Tin_manual[:,0]=12
for i in range(0,Appnum):
    for t in range(0, T-1):
        if Tin_manual[i,t]<Tset_manual[i,t]:
            Tin_manual[i,t+1]= Tin_manual[i,t]+(Tout[i,t]-Tin_manual[i,t])*z+(30-Tin_manual[i,t])*z
        else:
            Tin_manual[i,t+1]= Tin_manual[i,t]+(Tout[i,t]-Tin_manual[i,t])*z
                
k=4
major_ticks = np.arange(0, (k+1)*96, 96)
plt.plot(Tin_manual[0,0:96*k], 'ko-', linewidth=2, markersize=3)
plt.xticks(major_ticks)
plt.ylabel('Indoor Temperature (Manual Operation)')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 50))
plt.show()         

#%%
Cost_manual=np.zeros((Appnum,T))
for i in range(0,Appnum):
    for t in range(0, T-1):
        if Tin_manual[i,t]<Tset_manual[i,t]:
            
            Cost_manual[i,t]=Price[t]*np.absolute(30)+w[i,t]*np.absolute(Tdes[i,t]-Tin_manual[i,t])
        else:
            Cost_manual[i,t]=0*Price[t]*np.absolute(30)+w[i,t]*np.absolute(Tdes[i,t]-Tin_manual[i,t])
                
k=4
major_ticks = np.arange(0, (k+1)*96, 96)
plt.plot(Cost_manual[0,0:96*k], 'ko-', linewidth=2, markersize=3)
plt.xticks(major_ticks)
plt.ylabel('Cost (Manual Operation)')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 50))
plt.show()    
#%%
Normal=0.1
Tin=np.zeros((Appnum,T))
Tair=10*np.ones((Appnum,T))
Tset=np.zeros((Appnum,T))
u=np.zeros((Appnum,T))

probair=np.zeros((Appnum,NumSlice+1,T))
probset=np.zeros((Appnum,NumSlice,T))
State=np.zeros((Appnum,5,T))
StateRound=np.zeros((Appnum,5,T))


Loss=np.zeros((Appnum,T))
Losscheck=np.zeros((Appnum,T))

valuefun=np.zeros((Appnum,T))
delta=np.zeros((Appnum,T))
Lossv=torch.Tensor((Appnum,T))
Lossp=np.zeros((Appnum,T))
Cost=np.zeros((Appnum,T))
indicesair=np.zeros((Appnum,T))
indicesset=np.zeros((Appnum,T))


StartTime={}
DeadLine={}
Duration={}
Capacity={}
Pup={}
Plow={}
policy = {} 
policyold={}
valufunction={}
for i in range(0,Appnum):
   
    policy[i] = nnPolicy.Network(5, 12, [12,12,12])    
    checkpoint_policy = {'input_size': 5,
                          'output_size': 12,
                          'hidden_layers': [each.out_features for each in policy[i].hidden_layers],
                          'state_dict': policy[i].state_dict()}
    linkindex='checkpoint_pol'+str(i)+'.pth'
    torch.save(checkpoint_policy, linkindex)   
    policyold[i]=policy[i]
    
    
    valufunction[i] = nnValue.Network(5, 1, [10,10,10])    
    checkpoint_valufunction = {'input_size': 5,
                                  'output_size': 1,
                                  'hidden_layers': [each.out_features for each in valufunction[i].hidden_layers],
                                  'state_dict': valufunction[i].state_dict()}
    linkindex2='checkpoint_value'+str(i)+'.pth'
    torch.save(checkpoint_valufunction, linkindex2)
#%%
for t in range(0,T-1):
    print(t)
    for Appindex in range(0,Appnum):
        if t==0:
            Tin[Appindex,t]=12;
            State[Appindex,0,t]=Tin[Appindex,t];
            State[Appindex,1,t]=Tout[Appindex,t];
            State[Appindex,2,t]=Tdes[Appindex,t];
            State[Appindex,3,t]=N[Appindex,t];
            State[Appindex,4,t]=Price[t];

            
            linkindex='checkpoint_pol'+str(Appindex)+'.pth'
            policy[Appindex] = UsefulFunc.load_checkpointPolicy(linkindex)
            StateRound[Appindex,:,t]=np.around(State[Appindex,:,t],decimals=3)
            Soldpo=torch.from_numpy(StateRound[Appindex,:,t]).float().view(1,5)
            pa=policy[Appindex].forward(Normal*Soldpo)
            #pis=torch.Tensor.cpu(ps).detach().numpy()
            pia=torch.Tensor.cpu(pa).detach().numpy()
            #indicesset[Appindex,t]=np.random.choice(11, 1, p=pis[0,:])
            indicesair[Appindex,t]=np.random.choice(12, 1, p=pia[0,:])
            # print(pia)
            # print(indicesair)
            if indicesair[Appindex,t]==11:
                u[Appindex,t]=0
                Tair[Appindex,t]=0
            else:
                u[Appindex,t]=1
                Tair[Appindex,t]=Tairmin[Appindex,t]+PowerSlice*int(indicesair[Appindex,t])*(Tairmax[Appindex,t]-Tairmin[Appindex,t])
            #Tset[Appindex,t]=Tsetmin[Appindex,t]+PowerSlice*int(indicesset[Appindex,t])*(Tsetmax[Appindex,t]-Tsetmin[Appindex,t])
            #probset[Appindex,:,t]=pis[0,:]
            probair[Appindex,:,t]=pia[0,:]
            
        else:              
            StateRound[Appindex,:,t]=np.around(State[Appindex,:,t],decimals=3)
            S=torch.from_numpy(StateRound[Appindex,:,t]).float().view(1,5)
                  
            linkindex2='checkpoint_value'+str(Appindex)+'.pth'
            valufunction[Appindex] = UsefulFunc.load_checkpointValue(linkindex2)
            #optimizer = optim.Adam(valufunction[Appindex].parameters(), lr=0.1)
            optimizer = optim.Adagrad(valufunction[Appindex].parameters(), lr=2, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
            #optimizer = optim.Adadelta(valufunction[Appindex].parameters(), lr=2.0, rho=0.9, eps=1e-06, weight_decay=0)
            #optimizer = optim.SGD(valufunction[Appindex].parameters(), lr=0.002/(t**0.6))
            vsnew=valufunction[Appindex].forward(Normal*S)
            vnew=torch.Tensor.cpu(vsnew).detach().numpy()
            
            StateRound[Appindex,:,t-1]=np.around(State[Appindex,:,t-1],decimals=3)
            Sold=torch.from_numpy(StateRound[Appindex,:,t-1]).float().view(1,5)
            vsold=valufunction[Appindex].forward(Normal*Sold)
            vold=torch.Tensor.cpu(vsold).detach().numpy()
            
            delta[Appindex,t]=Cost[Appindex,t-1]+bet*vnew-vold
            Lossv=UsefulFunc.criterionvalue(vsnew, vsold, Cost[Appindex,t-1],bet, delta[Appindex,t])
            optimizer.zero_grad()
            Lossv.backward()
            optimizer.step()
            value=valufunction[Appindex].forward(Normal*S)
            valuefun[Appindex,t]=torch.Tensor.cpu(value).detach().numpy()
            checkpoint_Valufunction = {'input_size': 5,
                                       'output_size': 1,
                                       'hidden_layers': [each.out_features for each in valufunction[Appindex].hidden_layers],
                                       'state_dict': valufunction[Appindex].state_dict()}
            linkindex2='checkpoint_value'+str(Appindex)+'.pth'
            torch.save(checkpoint_Valufunction, linkindex2) 

            linkindex='checkpoint_pol'+str(Appindex)+'.pth'
            policy[Appindex] = UsefulFunc.load_checkpointPolicy(linkindex)
            #optimizerpolicy =optim.Adam(policy[Appindex].parameters(), lr=0.001)
            optimizerpolicy =optim.Adagrad(policy[Appindex].parameters(), lr=0.1 , lr_decay=0, weight_decay=0, initial_accumulator_value=0)
            #optimizerpolicy = optim.Adadelta(policy[Appindex].parameters(),lr=2.0, rho=0.9, eps=1e-06, weight_decay=0)
            #optimizerpolicy = optim.SGD(policy[Appindex].parameters(), lr=0.005)
            StateRound[Appindex,:,t-1]=np.around(State[Appindex,:,t-1],decimals=3)
            Soldpo=torch.from_numpy(StateRound[Appindex,:,t-1]).float().view(1,5)
            pairold=policy[Appindex].forward(Normal*Soldpo)
    
            Lossp=UsefulFunc.criterionpolicyCom(pairold[0,int(indicesair[Appindex,t-1])],pairold,delta[Appindex,t-1])
            Loss[Appindex,t]=torch.Tensor.cpu(Lossp).detach().numpy()
            optimizerpolicy.zero_grad()
            Lossp.backward()
    #               for param in policy.parameters():
    #                 print(param.grad.data)
            optimizerpolicy.step()
            checkpoint_policy = {'input_size': 5,
                                 'output_size': 12,
                                 'hidden_layers': [each.out_features for each in policy[Appindex].hidden_layers],
                                 'state_dict': policy[Appindex].state_dict()}
            linkindex='checkpoint_pol'+str(Appindex)+'.pth'
            torch.save(checkpoint_policy, linkindex)
            print(S*Normal)          
            pa=policy[Appindex].forward(Normal*S)
            pia=torch.Tensor.cpu(pa).detach().numpy()
        
            print(pia)
            
            indicesair[Appindex,t]=np.random.choice(12, 1, p=pia[0,:])
            if indicesair[Appindex,t]==11:
                u[Appindex,t]=0
                Tair[Appindex,t]=0
            else:
                u[Appindex,t]=1
                Tair[Appindex,t]=Tairmin[Appindex,t]+PowerSlice*int(indicesair[Appindex,t])*(Tairmax[Appindex,t]-Tairmin[Appindex,t])

            probair[Appindex,:,t]=pia[0,:]
            
            # print(probset)
            #print(probair)
        

        Tin[Appindex,t+1]= Tin[Appindex,t]+(Tout[Appindex,t]-Tin[Appindex,t])*z+(Tair[Appindex,t]-Tin[Appindex,t])*z*u[Appindex,t]
        Cost[Appindex,t]=u[Appindex,t]*Price[t]*np.absolute(Tair[Appindex,t])+N[Appindex,t]*w[Appindex,t]*np.absolute(Tdes[Appindex,t]-Tin[Appindex,t])
        State[Appindex,0,t+1]=Tin[Appindex,t+1]
        State[Appindex,1,t+1]=Tout[Appindex,t+1]
        State[Appindex,2,t+1]=Tdes[Appindex,t+1];
        State[Appindex,3,t+1]=N[Appindex,t+1];
        State[Appindex,4,t+1]=Price[t+1];
      

k=NumDays
major_ticks = np.arange(0, (k+1)*96, 96*5)
plt.plot(u[0,0:96*k], 'yo-', linewidth=1, markersize=2)
plt.xticks(major_ticks)
plt.ylabel('On/OFF (DRL)')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 500))
plt.show()


k=NumDays
major_ticks = np.arange(0, (k+1)*96, 96*5)
plt.plot(Tair[0,0:96*k], 'ro-', linewidth=1, markersize=2)
plt.xticks(major_ticks)
plt.ylabel('HVAC Air Temperature (DRL)')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 500))
plt.show()

k=NumDays
major_ticks = np.arange(0, (k+1)*96, 96*5)
plt.plot(Tin[0,0:96*k], 'ko-', linewidth=1, markersize=2)
plt.xticks(major_ticks)
plt.ylabel('Indoor Temperature (DRL)')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 500))
plt.show()




major_ticks = np.arange(0, (k+1)*96, 96*5)
plt.plot(valuefun[0,0:96*k], 'bo-', linewidth=1, markersize=2)
plt.xticks(major_ticks)
plt.ylabel('Value Function (DRL)')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 50))
plt.show()


major_ticks = np.arange(0, (k+1)*96, 96*5)
plt.plot(delta[0,0:96*k], 'bo-', linewidth=1, markersize=2)
plt.xticks(major_ticks)
plt.ylabel('TD Error (DRL)')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 500))
plt.show()     


major_ticks = np.arange((k-4)*96, (k+1)*96, 96*5)
plt.plot(Cost[0,(k-5)*96:96*k], 'ro-', linewidth=1, markersize=2)
plt.xticks(major_ticks)
plt.ylabel('Cost with DRL (cents)')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 500))
plt.show()


major_ticks = np.arange((k-4)*96, (k+1)*96, 96*5)
plt.plot(Cost_manual[0,(k-5)*96:96*k], 'go-', linewidth=1, markersize=2)
plt.xticks(major_ticks)
plt.ylabel('Cost with pre-scheduled setpoint (cents)')
plt.xlabel('Time Slot')
plt.figure(figsize=(30, 500))
plt.show()

D1=np.sum(Cost[0,(k-5)*96:96*(k-4)])
D2=np.sum(Cost[0,(k-4)*96:96*(k-3)])
D3=np.sum(Cost[0,(k-3)*96:96*(k-2)])
D4=np.sum(Cost[0,(k-2)*96:96*(k-1)])
D5=np.sum(Cost[0,(k-1)*96:96*(k)])
DRL=[D1,D2,D3,D4]
# creating the dataset
data = {'Day1':D1, 'Day2':D2, 'Day3':D3, 
        'Day4':D4, 'Day5':D5}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon', 
        width = 0.4)
 
plt.ylabel("Daily Cost ($)")
plt.title("Cost with DRL-based Autonomous Control")
plt.show()

D11=np.sum(Cost_manual[0,(k-5)*96:96*(k-4)])
D22=np.sum(Cost_manual[0,(k-4)*96:96*(k-3)])
D33=np.sum(Cost_manual[0,(k-3)*96:96*(k-2)])
D44=np.sum(Cost_manual[0,(k-2)*96:96*(k-1)])
D55=np.sum(Cost_manual[0,(k-1)*96:96*(k)])
manual=[D11,D22,D33,D44]

# creating the dataset
data = {'Day1':D11, 'Day2':D22, 'Day3':D33, 
        'Day4':D44, 'Day5':D55}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='blue', 
        width = 0.4)
 
plt.ylabel("Daily Cost ($)")
plt.title("Cost with Pre-scheduled Setpoint Control")
plt.show()

# plt.plot(Cost[0,T-2*D:T-1])
# plt.show()
# plt.plot(Cost[0,0:2*D])
# plt.show()
# plt.plot(P[0,T-2*D:T-1])
# plt.show()
# plt.plot(P[0,0:2*D])
# plt.show()
#P2=P[0,T-10*D:T-D]
#Pnew=P2.reshape(-1,D)
#s=np.sum(Pnew,axis=0)
#plt.plot(s/9)
#plt.show()
#           
#P2=P[0,0:9*D]
#Pnew=P2.reshape(-1,D)
#s=np.sum(Pnew,axis=0)
#plt.plot(s/10)
#plt.show()       
 
# for Appindex in range(0,Appnum):
#          P2=P[Appindex,T-5*D:T-D]
#          Pnew=P2.reshape(-1,D)
#          s=np.sum(Pnew,axis=0)  
#          stotalnew=stotalnew+s/4
           
# for Appindex in range(0,Appnum):
#          P2=PnDR[Appindex,T-5*D:T-D]
#          Pnew=P2.reshape(-1,D)
#          s=np.sum(Pnew,axis=0)  
#          stotalold=stotalold+s/4 
# tt = np.arange(0, D, 1)
# plt.plot(tt,stotalnew,tt, stotalold)
# plt.show()        
#plt.plot(stotalold)
#plt.show()
#plt.plot(stotalnew)
#plt.show()

