import numpy as np
import matplotlib.pyplot as plt



#####################################################################################
#Temperature data

"""
file = open('effTemp2006_10.txt','r')
temp_2006_10 = []
for line in file:
    temp_2006_10.append(float(line))
file.close()
temp=np.array(temp_2006_10)
"""
file = open('effTemp2005_10.txt','r')
temp_2005_10 = []
for line in file:
    temp_2005_10.append(float(line))
file.close()
temp=np.array(temp_2005_10)


"""
file = open('effTemp2001_17.txt','r')
temp_2001_17 = []
for line in file:
    temp_2001_17.append(float(line))
file.close()
temp=np.array(temp_2001_17)
"""




######################################################################################

a=6

year_names = []
year_starts = []
y=[]
u=[]
pre_gap=[]
post_gap=[]
c=0
for l in range(a):
	
	year_names.append(f"{l+5:02d}")
	if((l+1)%4==0):
		year_starts.append(365*(l+1)+1)
	else:
		year_starts.append(365*(l+1))





	file = open("eff_press_corrected_20"+f"{l+5:02d}"+".txt","r")
	print("eff_press_corrected_20"+f"{l+5:02d}"+".txt")
	x1 = []
	for p in file:
	    x1.append(float(p))
	file.close()

	file = open("eff_press_corrected_time_20"+f"{l+5:02d}"+".txt","r")
	print("eff_press_corrected_time_20"+f"{l+5:02d}"+".txt")
	t1 = []
	for p in file:
	    t1.append(float(p))
	file.close()
	
	t1=np.array(t1)
	x1=np.array(x1)
	#print(365/len(t1))
	#print(t1[1]-t1[0])
	#for k in range(len(t1)-4,len(t1)):
	#	print(t1[k])
	Mean=x1.mean()
	StD=x1.std()
	print(Mean,StD)

	sum=0.0
	count=0
	for i in range(len(t1)):
		if (Mean-3*StD<=x1[i]<=Mean+3*StD):
			sum=sum+x1[i]
			count=count+1
	#print(sum/count)
	
	for j in range(len(t1)):
		if x1[j]>0:
			x1[j]=x1[j]
			y.append(x1[j])
		else:
			x1[j]=sum/count
			y.append(sum/count)
		if (l+1)%4==0:
		
			u.append(c)
			c=c+366/len(t1)
		else:
			u.append(c)
			c=c+365/len(t1)
		#u.append(c)
		#c=c+t1[1]-t1[0]
		#print(c)
	sum1=0.0
	count1=0
	
	for k in range(0,360):
		sum1=sum1+x1[k]
		count1=count1+1
	pre_gap.append(sum1/count1)
	
	sum2=0.0
	count2=0
	for k in range(len(t1)-360,len(t1)):
		sum2=sum2+x1[k]
		count2=count2+1
	post_gap.append(sum2/count2)

plt.minorticks_on()
plt.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True, labelbottom=True)
plt.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)


plt.plot(u,y,'y')
plt.xlim(np.min(u),np.max(u))
plt.ylim(2000,3200)
plt.ylabel(r'Muon Rate (s$^{-1}$)',size='x-large')
plt.xlabel('Time(Year)')
plt.xticks(year_starts, year_names)
plt.grid(color='k',axis='x', linestyle='--', linewidth=1.2)
plt.show()

###############################################
M=[]	
for m in range(len(pre_gap)-1):
	M.append(pre_gap[m+1]/post_gap[m])
	print(pre_gap[m+1])
	print(post_gap[m])
M=np.array(M)


print(M)



A=[]
mul=1
A.append(1)
for n in range(len(M)):
	mul =mul*M[n]
	A.append(mul)
	print(mul)
print(A)		
	
#############################################################################################

y=[]
print(y)
u=[]
c=0
for l in range(a):
	
	
	

	file = open("eff_press_corrected_20"+f"{l+5:02d}"+".txt","r")
	print("eff_press_corrected_20"+f"{l+5:02d}"+".txt")
	x1 = []
	for p in file:
	    x1.append(float(p))
	file.close()

	file = open("eff_press_corrected_time_20"+f"{l+5:02d}"+".txt","r")
	print("eff_press_corrected_time_20"+f"{l+5:02d}"+".txt")
	t1 = []
	for p in file:
	    t1.append(float(p))
	file.close()
	
	t1=np.array(t1)
	x1=np.array(x1)
	#print(365/len(t1))
	#print(t1[1]-t1[0])
	#for k in range(len(t1)-4,len(t1)):
	#	print(t1[k])

	Mean=x1.mean()
	StD=x1.std()
	print(Mean,StD)
	
	sum=0.0
	count=0
	for i in range(len(t1)):
		if (Mean-3*StD<=x1[i]<=Mean+3*StD):
			sum=sum+x1[i]
			count=count+1
	#print(sum/count)

	for j in range(len(t1)):
		if x1[j]>0:
			x1[j]=x1[j]
			y.append(x1[j]/A[l])
		else:
			x1[j]=sum/count
			y.append((sum/count)/A[l])
		if (l+1)%4==0:
		
			u.append(c)
			c=c+366/len(t1)
		else:
			u.append(c)
			c=c+365/len(t1)



y=np.array(y)
u=np.array(u)

	
print(pre_gap,post_gap)			
#print(np.min(u),np.max(u))

plt.minorticks_on()
plt.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True, labelbottom=True)
plt.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)


plt.plot(u,y,'b')
plt.xlim(np.min(u),np.max(u))
plt.ylim(2000,3200)
plt.ylabel(r'Muon Rate (s$^{-1}$)',size='x-large')
plt.xlabel('Time(Year)',size='x-large')
plt.xticks(year_starts, year_names)
plt.grid(color='k',axis='x', linestyle='--', linewidth=1.2)
plt.show()
#np.save('muon_raw_data_2001_17',y)
#np.save('muon_time_raw_data_2001_17',u)

#############################################################################################

time =u
fn = y
const=0.125/2

count=0
muon_period=[]
time_period=[]

sum=0
counter=0

for i in range(len(time)):
	#sum=0
	#counter=0
	
	if(count-const<=time[i]<count+const):
		sum=sum+fn[i]
		counter=counter+1
	
	
	if(time[i]>=count+const):
		print(time[i])
		muon_period.append(sum/counter)
		time_period.append(count)
		sum=0
		counter=0
		count=count+0.125

print(muon_period,time_period)
print(len(muon_period))

plt.minorticks_on()
plt.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True, labelbottom=True)
plt.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)



plt.plot(time_period, muon_period,'r')
plt.xlim(np.min(time_period),np.max(time_period))
plt.ylabel(r'Muon Rate (s$^{-1}$)',size='x-large')
plt.xlabel('Time(Year)',size='x-large')
plt.xticks(year_starts, year_names)
plt.grid(color='k',axis='x', linestyle='--', linewidth=1.2)
plt.show()
##############################################################################
# Temperature correction
t=np.array(time_period)
mu=np.array(muon_period)


Mean=mu.mean()
StD=mu.std()
print(Mean,StD)

sum5=0.0
count5=0
for i in range(len(t)):
	if (Mean-3*StD<=mu[i]<=Mean+3*StD):
		sum5=sum5+mu[i]
		count5=count5+1
Muon_mean=sum5/count5

fn=temp


temp_Mean=fn.mean()
temp_StD=fn.std()
print(temp_Mean,temp_StD)

sum6=0.0
count6=0
for i in range(len(t)):
	if (temp_Mean-3*temp_StD<=fn[i]<=temp_Mean+3*temp_StD):
		sum6=sum6+fn[i]
		count6=count6+1
temp_mean=sum6/count6
temp_diff=(fn-temp_mean)





n1=len(t)
n2=len(mu)
n3=len(temp_diff)
print(n1,n2,n3)
R_bar=Muon_mean
alpha=-0.0016
R=[]
print(t[0],t[1],mu[0],mu[1],temp_diff[0],temp_diff[1])
for i in range(len(t)):
	R.append(mu[i]-alpha*R_bar*temp_diff[i])
#plt.plot(t,mu,'b',linewidth=0.4,alpha=1,label='Without temperature-correction')
plt.plot(t,R,'r',linewidth=0.4,alpha=0.9,label='Temperature-corrected')
plt.xlim(np.min(t),np.max(t))
plt.xticks(year_starts,year_names)
plt.ylim(2900,3150)
plt.ylabel(r'Muon rate(s$^{-1}$) ')
plt.xlabel('Time(Year)')
plt.legend(frameon=False)
plt.savefig('compared15.pdf')
plt.show()

plt.hist(mu,bins=4000,density=True)
mean=np.mean(mu)
plt.axvline(x=mean,color='r')
plt.hist(R,bins=4000,density=True,alpha=0.9)
mean1=np.mean(R)
plt.axvline(x=mean1,color='r')
#plt.xlim(3000,3150)
plt.show()
#np.savetxt('muon_data_temp_corrected2001_17.txt',R)
#np.savetxt('time_data_temp_corrected2001_17.txt',t)

	
	
	




############################################################################################

# Running average

time=np.array(time_period)
fn=np.array(muon_period)


Mean=fn.mean()
StD=fn.std()
print(Mean,StD)

sum5=0.0
count5=0
for i in range(len(time)):
	if (Mean-3*StD<=fn[i]<=Mean+3*StD):
		sum5=sum5+fn[i]
		count5=count5+1
Muon_mean=sum5/count5
muon_variation=(fn-Muon_mean)*100/Muon_mean

for i in range(len(time)):
	if (-10.0<=muon_variation[i]<=10.0):
		muon_variation[i]=muon_variation[i]
	else:
		muon_variation[i]=0.0

ft=muon_variation

T_min=np.min(time)
T_max=np.max(time)
delta_T=60
dt=time[1]-time[0]
muon_period=[]
time_period=[]
muon_percentage=[]
#mu_error=[]
while(T_min<=T_max):
	count=0
	sum=0.0
	sum1=0.0
	k=0.0
	for i in range(len(time)):
		if(T_min-delta_T/2<=time[i]<T_min+delta_T/2):
			sum=sum+fn[i]
			sum1=sum1+ft[i]
			k=k+fn[i]*fn[i]
			count=count+1
			
		
	muon_period.append(sum/count)
	muon_percentage.append(sum1/count)
	time_period.append(T_min)
	#mu_error.append(np.sqrt((k/count)-((sum*sum)/(count*count)))/np.sqrt(count))
	print(T_min)
	T_min=T_min+dt
	
	
	#print(muon_period,time_period,mu_error)	
#plt.errorbar(time_period, muon_period,mu_error)
#plt.ylim(3000,3200)
#plt.xlim(0,20)
print(time_period)
plt.minorticks_on()
plt.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True, labelbottom=True)
plt.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)


plt.plot(time_period,muon_period,'k')
plt.xlim(np.min(time_period),np.max(time_period))
plt.ylabel(r'Muon Rate (s$^{-1}$)',size='x-large')
plt.xlabel('Time(Year)',size='x-large')
plt.xticks(year_starts, year_names)
plt.grid(color='k',axis='x', linestyle='--', linewidth=1.2)
plt.show()

#print(time_period)
plt.minorticks_on()
plt.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True, labelbottom=True)
plt.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)


plt.plot(time_period,muon_percentage,'g')
plt.xlim(np.min(time_period),np.max(time_period))
plt.ylabel('Muon variation (%)',size='x-large')
plt.xlabel('Time(Year)',size='x-large')
plt.xticks(year_starts, year_names)
plt.grid(color='k',axis='x', linestyle='--', linewidth=1.2)
plt.show()

###########################################################################################

# Running average

time=np.array(time_period)
fn=temp


Mean=fn.mean()
StD=fn.std()
print(Mean,StD)

sum5=0.0
count5=0
for i in range(len(time)):
	if (Mean-3*StD<=fn[i]<=Mean+3*StD):
		sum5=sum5+fn[i]
		count5=count5+1
Muon_mean=sum5/count5
muon_variation=(fn-Muon_mean)

for i in range(len(time)):
	if (-10.0<=muon_variation[i]<=10.0):
		muon_variation[i]=muon_variation[i]
	else:
		muon_variation[i]=0.0

ft=muon_variation

T_min=np.min(time)
T_max=np.max(time)
delta_T=60
dt=time[1]-time[0]
temp_period=[]
time_period=[]
temp_difference=[]
#mu_error=[]
while(T_min<=T_max):
	count=0
	sum=0.0
	sum1=0.0
	k=0.0
	for i in range(len(time)):
		if(T_min-delta_T/2<=time[i]<T_min+delta_T/2):
			sum=sum+fn[i]
			sum1=sum1+ft[i]
			k=k+fn[i]*fn[i]
			count=count+1
			
		
	temp_period.append(sum/count)
	temp_difference.append(sum1/count)
	time_period.append(T_min)
	#mu_error.append(np.sqrt((k/count)-((sum*sum)/(count*count)))/np.sqrt(count))
	print(T_min)
	T_min=T_min+dt
	
	
	#print(muon_period,time_period,mu_error)	
#plt.errorbar(time_period, muon_period,mu_error)
#plt.ylim(3000,3200)
#plt.xlim(0,20)
print(time_period)
plt.minorticks_on()
plt.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True, labelbottom=True)
plt.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)


plt.plot(time_period,temp_period,'k')
plt.xlim(np.min(time_period),np.max(time_period))
plt.ylabel(r'Muon Rate (s$^{-1}$)',size='x-large')
plt.xlabel('Time(Year)',size='x-large')
plt.xticks(year_starts, year_names)
plt.grid(color='k',axis='x', linestyle='--', linewidth=1.2)
plt.show()

plt.minorticks_on()
plt.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True, labelbottom=True)
plt.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)


plt.plot(time_period,temp_difference,'g')
plt.xlim(np.min(time_period),np.max(time_period))
plt.ylabel('Muon variation (%)',size='x-large')
plt.xlabel('Time(Year)',size='x-large')
plt.xticks(year_starts, year_names)
plt.grid(color='k',axis='x', linestyle='--', linewidth=1.2)
plt.show()

#################################################################################
# Fast Fourier Transformation

time = np.array(time_period)
fn = np.array(muon_percentage)
n = len(time)
print(n)

dx = time[1] - time[0]
karr = np.empty(int(n/2)+1, dtype=np.float64)
for m in range(int(n/2)+1):
    karr[m] = m/((n-1)*dx)
    
    
dft= np.empty(n, dtype=complex)
P = np.empty(int(n/2+1), dtype=np.float64)

#dft
dft = np.fft.fft(fn)
dft = dx*dft/(time[-1]-time[0])
            
#power spectrum using Periodogram estimate
P[0] = (np.abs(dft[0]))**2
for p in range(int((n/2)-1)):
    P[p+1] = (np.abs(dft[p+1])**2 + np.abs(dft[n-p-1])**2)/2.0
P[int(n/2)] = (np.abs(dft[int(n/2)]))**2
P = (time[-1]-time[0])*P

#print(P)

P = karr*P/np.pi

temp = 0.0
for i in range(int(n/2)+1):
    if (P[i] > temp):
        temp = P[i]
        w1 = i
    
f_c = karr[5]
print(f_c)
#delta_f = (karr[1] - karr[0])*1.25
delta_f=0.00035;
print(delta_f)
PS_filter = []

for q in range(int((n/2)+1)):
    if(np.abs(karr[q])>=f_c-delta_f and np.abs(karr[q])<=f_c+delta_f):
        PS_filter.append(P[q])
    elif((np.abs(karr[q])>f_c-2*delta_f and np.abs(karr[q])<f_c-delta_f) or (np.abs(karr[q])>f_c+delta_f and np.abs(karr[q])<f_c+2*delta_f)):
        fact = np.sin((np.pi*np.abs((np.abs(karr[q])-f_c)))/(2*delta_f))
        PS_filter.append(fact*P[q])
    else:
        PS_filter.append(0.0)

plt.plot(karr, P, label='PS')
plt.plot(karr, PS_filter, label='filtered PS')
plt.xlim(0, 0.012)
#plt.ylim(0, 20)
#plt.set_xscale('log')
plt.ylabel("PS of muon intensity variation(%)")
plt.xlabel("frequency(CPD)")
plt.legend()
plt.savefig('1MUONRATE_PS.pdf')

plt.show()

############################################################################################
# IFFT 


time = np.array(time_period)
fn = np.array(muon_percentage)
n = len(time)
print(n)

dt = time[1] -time[0]
#computing ps
#dft = np.abs(np.fft.fft(fn))**2/n
dft = np.fft.fft(fn)

#k array
warr = np.fft.fftfreq(n, d = dt)
f_c = warr[5]
#delta_f=(warr[7]-warr[6])*1.25
delta_f = 0.00035
#delta_f=0.0005
print(delta_f)
t_filter = []
data_filter = []

dft_filtered = []

for q in range(n):
    if(np.abs(warr[q])>=f_c-delta_f and np.abs(warr[q])<=f_c+delta_f):
        dft_filtered.append(dft[q])
    elif((np.abs(warr[q])>f_c-2*delta_f and np.abs(warr[q])<f_c-delta_f) or (np.abs(warr[q])>f_c+delta_f and np.abs(warr[q])<f_c+2*delta_f)):
        fact =np.sin((np.pi*np.abs((np.abs(warr[q])-f_c)))/(2*delta_f))
        dft_filtered.append((np.sqrt(fact))*dft[q])
    else:
        dft_filtered.append(0.0)
            
#plt.plot(warr, dft_filtered)
#plt.xlim(-0.012,0.012)

dw = warr[1] - warr[0]
print(dw)
time_filtered = np.fft.fftfreq(n, d = dw)

data_filtered = np.fft.ifft(dft_filtered)
muon_data_filtered = data_filtered.real
print(data_filtered)

plt.plot(time, muon_data_filtered,'r')
plt.xlim(np.min(time),np.max(time))
#plt.ylim(-0.3,0.3)
plt.ylabel("Muon intensity variation(%)")
plt.xlabel("Time(days)")
plt.savefig('IFFT_mu.pdf')

plt.show()




####################################################################################

# Fast Fourier Transformation

time = np.array(time_period)
fn = np.array(temp_difference)
n = len(time)
print(n)

dx = time[1] - time[0]
karr = np.empty(int(n/2)+1, dtype=np.float64)
for m in range(int(n/2)+1):
    karr[m] = m/((n-1)*dx)
    
    
dft= np.empty(n, dtype=complex)
P = np.empty(int(n/2+1), dtype=np.float64)

#dft
dft = np.fft.fft(fn)
dft = dx*dft/(time[-1]-time[0])
            
#power spectrum using Periodogram estimate
P[0] = (np.abs(dft[0]))**2
for p in range(int((n/2)-1)):
    P[p+1] = (np.abs(dft[p+1])**2 + np.abs(dft[n-p-1])**2)/2.0
P[int(n/2)] = (np.abs(dft[int(n/2)]))**2
P = (time[-1]-time[0])*P

#print(P)

P = karr*P/np.pi

temp = 0.0
for i in range(int(n/2)+1):
    if (P[i] > temp):
        temp = P[i]
        w1 = i
    
f_c = karr[5]
print(f_c)
#delta_f = (karr[1] - karr[0])*1.25
delta_f=0.00035;
print(delta_f)
PS_filter = []

for q in range(int((n/2)+1)):
    if(np.abs(karr[q])>=f_c-delta_f and np.abs(karr[q])<=f_c+delta_f):
        PS_filter.append(P[q])
    elif((np.abs(karr[q])>f_c-2*delta_f and np.abs(karr[q])<f_c-delta_f) or (np.abs(karr[q])>f_c+delta_f and np.abs(karr[q])<f_c+2*delta_f)):
        fact = np.sin((np.pi*np.abs((np.abs(karr[q])-f_c)))/(2*delta_f))
        PS_filter.append(fact*P[q])
    else:
        PS_filter.append(0.0)

plt.plot(karr, P, label='PS')
plt.plot(karr, PS_filter, label='filtered PS')
plt.xlim(0, 0.012)
#plt.ylim(0, 20)
#plt.set_xscale('log')
plt.ylabel("PS of muon intensity variation(%)")
plt.xlabel("frequency(CPD)")
plt.legend()
plt.savefig('1MUONRATE_PS.pdf')

plt.show()

##############################################################################################
# IFFT 


time = np.array(time_period)
fn = np.array(temp_difference)
n = len(time)
print(n)

dt = time[1] -time[0]
#computing ps
#dft = np.abs(np.fft.fft(fn))**2/n
dft = np.fft.fft(fn)

#k array
warr = np.fft.fftfreq(n, d = dt)
f_c = warr[5]
#delta_f=(warr[7]-warr[6])*1.25
delta_f = 0.00035
#delta_f=0.0005
print(delta_f)
t_filter = []
data_filter = []

dft_filtered = []

for q in range(n):
    if(np.abs(warr[q])>=f_c-delta_f and np.abs(warr[q])<=f_c+delta_f):
        dft_filtered.append(dft[q])
    elif((np.abs(warr[q])>f_c-2*delta_f and np.abs(warr[q])<f_c-delta_f) or (np.abs(warr[q])>f_c+delta_f and np.abs(warr[q])<f_c+2*delta_f)):
        fact =np.sin((np.pi*np.abs((np.abs(warr[q])-f_c)))/(2*delta_f))
        dft_filtered.append((np.sqrt(fact))*dft[q])
    else:
        dft_filtered.append(0.0)
            
#plt.plot(warr, dft_filtered)
#plt.xlim(-0.012,0.012)

dw = warr[1] - warr[0]
print(dw)
time_filtered = np.fft.fftfreq(n, d = dw)

data_filtered = np.fft.ifft(dft_filtered)
temp_data_filtered = data_filtered.real
print(data_filtered)

plt.plot(time, temp_data_filtered,'r')
plt.xlim(np.min(time),np.max(time))
#plt.ylim(-0.3,0.3)
plt.ylabel("Muon intensity variation(%)")
plt.xlabel("Time(days)")
plt.savefig('IFFT_mu.pdf')

plt.show()

########################################################################################
# Temperature coefficient


mu = np.array(muon_data_filtered)
temp = np.array(temp_data_filtered)
print(mu)
print(temp)
print(np.min(temp))
print(np.max(temp))
x = np.min(temp)
print(x)
temp_binned = []
muon_binned = []
mu_error=[]
while(x<=np.max(temp)):
	count = 0
	y = 0.0
	z = 0.0
	k = 0.0	
	for i in range(len(mu)):
		if(temp[i]>=x and temp[i]<x+0.05):
			y = y + temp[i]
			z = z + mu[i]
			k=k+mu[i]*mu[i]
			count = count + 1 
	#temp_binned.append(y/count)
	temp_binned.append(x)
	muon_binned.append(z/count)
	mu_error.append(np.sqrt((k/count)-((z*z)/(count*count)))/np.sqrt(count))
	x = x + 0.05 
print(temp_binned)
print(muon_binned)
print(mu_error)
plt.plot(temp_binned,muon_binned,'.k')
plt.show()

######################################################################################
#MCMC

from scipy.optimize import minimize

import emcee

import corner



#ln L function:

def log_likelihood(theta,x,y,yerr):

    a,b = theta

    model = a*x+b

    sigma2 = yerr**2

    return(0.5*np.sum((y-model)**2/sigma2+np.log(2*np.pi*sigma2)))       # actually negative ln L

 



#prior distribution p(m; b|I).

#We are a priori ignorant about the parameters so we choose a uniform prior    

def log_prior(theta):

    a,b = theta

    if(-500.0<a<500 and -500.0<b<500.0):

        return(0.0)

    return(-np.inf)



#posterior PDF:    

def log_probability(theta,x,y,yerr):

    lp = log_prior(theta)

    if not(np.isfinite(lp)):

        return(-np.inf)

    return(lp-log_likelihood(theta,x,y,yerr))



#reading the data and storing them in arrays 

"""file=open('data.txt','r').read().split('\n')          #the file 'data.txt' contains the given datas 

#del file[0:5]     #deleting the first 5 lines which contain no data 

x=[]

y=[]

yerr=[]

for line in file:

    index,x_data,y_data,yerr_data=line.split('&')

    x.append(float(x_data))

    y.append(float(y_data))

    yerr.append(float(yerr_data))



x=np.array(x)

y=np.array(y)

yerr=np.array(yerr)

"""



x1 = np.array(temp_binned)

y1 = np.array(muon_binned)

yerr1 = np.array(mu_error)



x = []

y = []

yerr = []



for i in range(len(x1)):

    if(i>8 and i<(len(x1)-10)):

        x.append(x1[i])

        y.append(y1[i])

        yerr.append(yerr1[i])



x = np.array(x)

y = np.array(y)

yerr = np.array(yerr)



#Now we can sample our posterior PDF using MCMC.

#Let us use 50 Markov chains.

#Where do we initialise them? Anywhere we want but a common idea to start near the optimum of the likelihood.

guess = (1.0,1.0)

soln = minimize(log_likelihood,guess,args=(x,y,yerr))



#We now initialise each of our 50 Markov chains near the optimum reported by the minimize function

nwalkers, ndim = 32, 2

pos = soln.x+1e-4*np.random.randn(nwalkers,ndim)



#We now use the emcee library to do the MCMC so that each Markov chain takes 4,000 steps.

sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability,args=(x,y,yerr))

sampler.run_mcmc(pos,5000)



#We can look at the chains by plotting them:

samples = sampler.get_chain()



#best-fit values of the parameters

a_true = np.median(samples[:,:,0])

b_true = np.median(samples[:,:,1])



print('best-fit values of the parameters a, b are respectively',a_true,',',b_true)



#one-sigma uncertainty is given by standard deviation

print('one-sigma uncertainties are respectively',np.std(samples[:,:,0]),',',np.std(samples[:,:,1]))



"""

plt.plot(samples[:,:,0],'k') # a values

plt.xlabel(r'step',fontsize=20)

plt.ylabel(r'a',fontsize=20)

plt.title('Markov Chains for the parameter a',fontsize=20)

plt.show()



plt.plot(samples[:,:,1],'k') # b values

plt.xlabel(r'step',fontsize=20)

plt.ylabel(r'b',fontsize=20)

plt.title('Markov Chains for the parameter b',fontsize=20)

plt.show()

"""



#We can plot the posterior PDF using the corner library.

params=np.vstack([samples[i] for i in range(len(samples))])

fig=corner.corner(params,labels=['a','b'],truths=[a_true,b_true],show_titles=False)

plt.show()



fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 7

fig_size[1] = 7

plt.rcParams["figure.figsize"] = fig_size

plt.minorticks_on()

plt.tick_params('both', which='minor', length=2, width=1, direction='out', top=True, right=True)

plt.tick_params('both', which='major', length=4, width=1, direction='out', top=True, right=True)



#plotting the data with error bars

plt.errorbar(x1,y1,yerr=yerr1,fmt='.k',capsize=5)




z=np.linspace(x[0],x[-1],100)



"""

sample_model=np.random.randint(0,nwalkers*4000,200)



#plotting 200 randomly chosen models

for j in range(200):

    plt.plot(z,params[sample_model[j]][0]*z*z+params[sample_model[j]][1]*z+params[sample_model[j]][2],'silver')

"""



#plotting best fit model

plt.plot(z,a_true*z+b_true,'r')



#plt.xlabel(r'$x$',fontsize=20)

#plt.ylabel(r'$y$',fontsize=20)
plt.ylabel("Muon intensity Variation(%)")
plt.xlabel("Temperature change(K)")
plt.text(0.1,0.1,'Temp coeff')
plt.text(0.8,0.1,'1 sigma')
plt.text(0.1,0.07,np.round(a_true,4))
plt.text(0.8,0.07,np.round(np.std(samples[:,:,0]),4))

#plt.title(r'data with the best-fit model',fontsize=20)

plt.legend(frameon=False)

plt.savefig('alpha511_fft.pdf')

plt.show()

##########################################################################################################################

















