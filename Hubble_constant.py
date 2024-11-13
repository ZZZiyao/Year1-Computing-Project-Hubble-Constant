# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 00:05:24 2022

@author: xziyac
"""
#import essential packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy.solvers import solve
from sympy import Symbol


#load intensity and frequency data
data=np.loadtxt("D:\Data(6)\Data\Halpha_spectral_data.csv",skiprows=3,delimiter=',', unpack=True)
#load distance data
distance_mpc=np.loadtxt("D:\Data(6)\Data\Distance_Mpc.txt",skiprows=2,delimiter='\t', unpack=True)
print(data) #just for checking
print(distance_mpc)
#%%
# generate a list with observation number in first row of Halpha data
observation_number=[]
for i in data[range(0,len(data),2)]:  #since the number is same in every two elements
    observation_number.append(i[0])
    
print(observation_number) #for reference


# name the elements (every column) in Distance_Mpc data
obser=distance_mpc[0]
dis=distance_mpc[1]
res=distance_mpc[2]

#generate lists to input data after rearrangement, ie. the observation number of distance_mpc matches Halpha data

dist=[]
response=[]
#let every columns of data follows the order of corresponding observation number

for i in range(len(observation_number)):
    for j in range(len(observation_number)):
      if observation_number[i]==obser[j]:
        dist.append(dis[j])
        response.append(res[j])

print("the distance after rearrangement is",dist)#after rearrangement
print(response)

#we only care about sets of data which their corresponding responses==1
distance1=[]
for i in range(len(response)):
    if response[i]==1:
        distance1.append(dist[i])
        
print(distance1)
print(len(distance1))#to see how many values are remaining

#%%

#generate lists to input observed frequency and their error
obserfreq=[]
error=[]


for i in range(len(response)):  
  if int(response[i])==1:
      #again, we only care about the data if its response is good. Since now the responce are in the same order as Halpha data, we could use the same i for them.
    
    k='observation number:'+str(observation_number[i])#a string can be used as title of each plot
    
    x1=data[2*i]#column no. 0,2,4,6,8,,,,etc is our x data (frequency)
    y1=data[2*i+1]#column no. 1,3,5,7,9,,,,etc is our y data(intensity)
    x=x1[1:len(x1)]#we ignore the first data(x1[0])since that is an observation number
    y=y1[1:len(y1)]#same reason as above
    
    
    #split x and y to 20 sublists
    xsplits = np.array_split(x, 20)
    ysplits=np.array_split(y,20)

    #look for average value in each sublists
    #this is because the noise in initial data is too large, thus using several averges to fit curve is more reliable
    avef=[]
    aveinten=[]
    for array in xsplits:
        a=np.mean(array)
        avef.append(a)
        
    for array in ysplits:
        b=np.mean(array)
        aveinten.append(b)
    
      
    #define the fit function, with both linear part and the gausssian function
    #refernce:final exercise of core worksheet 2
    def fitting(x,a,mu,sig,m,c):
        gaus = a*np.exp(-(x-mu)**2/(2*sig**2))
        line = m*x+c
        return gaus + line
    
    
    #guess the parametres for each graph
    #a is max distance from straight line
    #mu is x coordinate of max y value of gaussian curve
    #m is gradient of straight line
    #c is y-intercept of straight line
    #the point with max y value is the point of furthest distance to the straight line
    guess_m=(aveinten[19]-aveinten[0])/(avef[19]-avef[0]) #gradient=(y2-y1)/(x2-x1)
    
    #y=mx+c,c=y-mx
    guess_c=aveinten[0]-guess_m*avef[0]
    
    #distance from point to line
    distance=[]
    for i in range(len(aveinten)):
      dis=aveinten[i]-guess_m*avef[i]#mx is the value of y if the straight line passes the point, thus distance is y coordinate of this point substracts this value
      distance.append(dis)
      
    position=np.argmax(distance)#find position of point with maximum distance

    guess_a=aveinten[position]-(aveinten[position]-guess_m*avef[position])#max distance
    guess_mu=avef[position]#x coordinate of that point is average of gaussian function
    guess_sig=avef[position+1]-guess_mu#standard deviation is (very) roughly equal to the separation between the point in the middle of gaussian function and the point next to it
    
    
    initial_guess=[guess_a,guess_mu,guess_sig,guess_m,guess_c]#initial guess of parametres

    popt,pcov=curve_fit(fitting,avef,aveinten,initial_guess,maxfev=10000) 
    #fit the function
  


    obserfreq.append(popt[1]) #since mu is the second parametre
    error.append(np.sqrt(pcov[1,1]))#the [1,1]element in the pcov matrix gives uncertainty in popt[1],ie.mu

    
    
#plot the graphs with the fitted curve 
    plt.plot(x,y,'.',label="given dataset",color='mediumaquamarine')  #inial dataset    
    plt.plot(avef,aveinten,'.',markersize=15,label="selected data points",color='darkgreen')#these are the points we actually used to fit the curve
    plt.plot(x,fitting(x,popt[0],popt[1],popt[2],popt[3],popt[4]),label='Fitted curve',linewidth=3,color='goldenrod')#fitted curve
    
    
    

    plt.xlabel("frequency(Hz)")
    plt.ylabel("Intensity (arb. unit)")
    
    
    plt.title(k)
              
    plt.legend()
   
    plt.show()

    
  
#%%
#convert lists into arrays
obserfrequency=np.array(obserfreq)
err=np.array(error)
percentage_error=err/obserfrequency#find percentage (actually fraction?) uncertainty in each observed frequency
print(percentage_error)

#%%
#finding velocity for each frequency
lambdae=656.28e-9#656.28nm
c=2.9979e8 #speed of light
#generate a list to input velocities
shift_v=[]
for i in obserfreq:
  v=Symbol('v') 
  #solve the function λ0/λe=sqrt((1+(v/c))/(1-(v/c)))
  #where λ0=c/frequency, c and λe are constants
  velocity=solve(((c/i)/lambdae)-(((1+(v/c))/(1-(v/c)))**0.5),v)
  a=float(velocity[0])
  shift_v.append(a/1000)#since we want velocity in km/s rather than m/s

print(shift_v)
print(len(shift_v))#for checking whether this matches length of distance1


#%%
#fit a straight line with diatance and velocity, as V=H0*D, the gradient is our hubble constant
#percentage error in velocity should be four times of that in frequency, since frequency is in quadratic order compares to velocity if we rearrange the equation describing the relationship between the velocity of the galaxy and the shift in wavelengths
#between the velocity of the galaxy and the shift in wavelengths. f^2 appears in both numerator and denominator, 2+2=4.
verr=4*percentage_error*np.array(shift_v)
fit,cov = np.polyfit(distance1,shift_v,1,w=1/verr,cov=True)
#fit[0]gives us the gradient of the best fit line

#%%

fitc=np.poly1d(fit)
plt.plot(distance1,fitc(distance1),color='olivedrab')#plot the straight line

errH = np.sqrt(cov[0,0]) #the uncertainty of the gradient would be [0,0] element of the covariance matrix


print("Estimate of Hubble's constant: %.4e +/- %.4e km/s/Mpc" %(fit[0],errH))
# print hubble constant we get with 4 decimal places in scientific notation.

#plot v & d we got from previous code and their errorbars
plt.errorbar(distance1,shift_v,yerr=verr,fmt='x',capsize=2,color='mediumseagreen')#as we can see, the uncertainty in each v value is really small that does not even appear on the graph
plt.xlabel('Distance(Mpc)',color='olive')
plt.ylabel('Redshift(km/s)',color='olive')
plt.title('Plot of redshift velocity against distance',color='olive')
plt.grid(color='khaki')
plt.text(150,24000,'Hubble constant = \n %.1f ± %.1f' %(fit[0],errH)+' km/s/Mpc',color='olive',fontsize=12)

#making the graph more beautiful, reference:https://www.skytowner.com/explore/changing_the_color_of_axes_in_matplotlib
ax = plt.gca()
ax.xaxis.label.set_color('olive')        
ax.yaxis.label.set_color('olive')          
ax.tick_params(axis='x', colors='olive')    
ax.tick_params(axis='y', colors='olive')  
ax.spines['left'].set_color('olive')        
ax.spines['bottom'].set_color('olive')  
ax.spines['top'].set_color('olive')   
ax.spines['right'].set_color('olive') 

plt.show()
#%%
A=np.array([[1,2,3],[4,5,6]])
print(A)
print(A[0,:])



