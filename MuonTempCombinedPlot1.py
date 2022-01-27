import numpy as np
#import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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

year_names = []
year_starts = []
for i in range(0,25,5):
	year_names.append(f"{2000+i:04d}")
	year_starts.append(365*(i+3))
#print(year_names,year_starts)

for i in range(140,150):
	print(i,ssn_time[i],ssn[i])
print(np.max(t2)+1440)
	


#year_starts =[0,1*365,2*365,2*365+366,3*365+366,4*365+366]
#year_names = ['2006','2007','2008','2009','2010','2011']

nplots = 2

#fig = plt.figure(figsize=(7, 7), dpi=100)
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
fig.subplots_adjust(hspace=0.0)
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

#--- Panel 1 

ax = fig.add_subplot(gs[:-1,:])
plt.minorticks_on()
ax.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True, labelbottom=False)
ax.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)

plt.plot(t1, mu,lw=1, zorder=-1, color='r')

ax.set_xlim(np.min(t1), np.max(t1))

#ax.set_xlim(2900,3200)
#ax.set_ylim(3050,3150)
ax.set_ylabel('Monthly mean SSN',size='x-large')

ax.set_xticks(year_starts)
ax.set_xticklabels(year_names)
plt.grid(color='k',axis='x', linestyle='--', linewidth=2)

"""plt.text(0.008,0.055, 'freq (CPD)')

plt.text(0.0105,0.055, 'year')

plt.text(0.008,0.05, np.round(t[1],7))
plt.text(0.008,0.045, np.round(t[4],7))
plt.text(0.008,0.04, np.round(t[9],7))
plt.text(0.008,0.035, np.round(t[15],7))

plt.text(0.0105,0.05, np.round(1/(t[1]*365),3))
plt.text(0.0105,0.045, np.round(1/(t[4]*365),3))
plt.text(0.0105,0.04, np.round(1/(t[9]*365),3))
plt.text(0.0105,0.035, np.round(1/(t[15]*365),3))
"""
#--- Panel 2

ax = fig.add_subplot(gs[-1,:])
plt.minorticks_on()
ax.tick_params('both', which='major', length=6, width=1, direction='in', top=True, right=True)
ax.tick_params('both', which='minor', length=4, width=1, direction='in', top=True, right=True)

plt.plot(muon_time, muon, lw=1, zorder=-1, color='g')
ax.set_xlim(np.min(t1), np.max(t1))
#ax.set_xlim(0,0.012)
#ax.set_ylim(215,219)
ax.set_ylabel(r'Muon rate ',size='x-large')
ax.set_xlabel('Time(Year)')

ax.set_xticks(year_starts)
ax.set_xticklabels(year_names)

plt.grid(color='k',axis='x', linestyle='--', linewidth=2)


plt.savefig('ssn_muon_data_1997_2020.pdf', bbox_inches='tight')
plt.show()
plt.close('All')
