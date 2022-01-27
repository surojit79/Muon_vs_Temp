import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import pearsonr
from scipy import stats


#reading the data
file_mu = open('ssn_data_monthly_avg_1997_2020.txt','r')
mu = []
for p in file_mu:
    mu.append(float(p))
file_mu.close()

file_temp = open('muon_data_monthly_avg_2001_2017.txt','r')
temp = []
for q in file_temp:
    temp.append(float(q))
file_temp.close()

file_time = open('ssn_time_data_monthly_avg_1997_2020.txt','r')
t1 = []
for r in file_time:
    t1.append(float(r))
file_time.close()

file_time1 = open('time_data_monthly_avg_2001_2017.txt','r')
t2 = []
for r in file_time1:
    t2.append(float(r))
file_time1.close()


ssn=np.array(mu)
ssn_time=np.array(t1)
muon=np.array(temp)
muon_time=np.array(t2)+1440


#cycle 24 starts from december 2008 i.e ssn_time=4320
#print(np.min(muon_time))
tau=-25*30
#T_min=2900
T_min=np.min(muon_time)
#T_max=np.max(muon_time)
T_max=4290

tau1=[]
corr1=[]
rho1=[]
pval1=[]



muon_cycle24=[]
for j in range(len(muon_time)):
	if T_min<=muon_time[j]<=T_max:
		muon_cycle24.append(muon[j])
		#print(j)

while(tau<=25*30):
	
	ssn_cycle24=[]
	for i in range(len(ssn_time)):
		if T_min+tau<=ssn_time[i]<=T_max+tau:
			ssn_cycle24.append(ssn[i])
			
	corr, _ = pearsonr(muon_cycle24,ssn_cycle24)
	corr1.append(corr)
	tau1.append(tau/30)
	rho, pval = stats.spearmanr(muon_cycle24,ssn_cycle24)
	rho1.append(rho)
	tau=tau+1*30
print(tau1)
print(corr1)

"""plt.plot(tau1,corr1,'--r')


plt.ylim(0.5,-1)
plt.text(18,-0.8,'Cycle 24',size='xx-large')
plt.show()"""

plt.plot(tau1,rho1,'--og')



plt.ylim(0,-1)
plt.xlim(-30,30)
plt.text(15,-0.8,'Cycle 23',size='xx-large')
plt.ylabel('Correlation coefficient',size='xx-large')
plt.xlabel('Time lag [months]',size='xx-large')
plt.minorticks_on()
plt.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True, labelbottom=True)
plt.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)
plt.savefig('Spearman_cycle23.pdf')
plt.show()
	
		
		


