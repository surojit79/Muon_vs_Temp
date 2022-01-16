import numpy as np
import matplotlib.pyplot as plt

time = np.arange(0,52584, 3)
t_min=np.min(time)
t_max=np.max(time)

effT = []

LeapYr = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
nonLeapYr = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

x = np.array([775, 750, 725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40, 30, 20, 10])
dx1 = []
#print(x[:-1], x[1:])
dx1 = x[:-1] - x[1:]
dx = np.zeros(len(x))
for p in range(len(x)-1):
	dx[p] = dx1[p]
dx[-1] = x [-1]
#print(dx)

const = 120.0
list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 345, 346, 347, 348]


h=8*365
year_starts =[0,h,2*h,3*h,4*h+24,5*h,6*h,7*h,8*h+2*24,9*h,10*h,11*h,12*h+3*24,13*h,14*h,15*h,16*h+4*24,17*h,18*h,19*h,20*h+5*24]
#year_names = ['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
year_names = []
year_starts = []

days = 0
for l in range(4,10):
	if((l+1)%4==0):
		monthDays = LeapYr
	else:
		monthDays = nonLeapYr
	
	year_names.append(f"{l+1:02d}")

	for m in range(12):
		for n in range(monthDays[m]):
			  
			#list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 345, 346, 347, 348]   

			#time = [0, 3, 6, 9, 12, 15, 18, 21]

			T = []
			
			if ((l+1)<=10):
				f = open("MERRA2_300.inst3_3d_asm_Np.20"+f"{l+1:02d}"+f"{m+1:02d}"+f"{n+1:02d}"+".nc4.ascii?PHIS[0:7][203:203][411:411],T[0:7][0:41][203:203][411:411],time,lat[203:203],lon[411:411],lev","r")
				print("MERRA2_300.inst3_3d_asm_Np.20"+f"{l+1:02d}"+f"{m+1:02d}"+f"{n+1:02d}"+".nc4.ascii?PHIS[0:7][203:203][411:411],T[0:7][0:41][203:203][411:411],time,lat[203:203],lon[411:411],lev")
			if((l+1)>10):
				if((l+1)==20 and (m+1)==9):
					f = open("MERRA2_401.inst3_3d_asm_Np.20"+f"{l+1:02d}"+f"{m+1:02d}"+f"{n+1:02d}"+".nc4.ascii?PHIS[0:7][203:203][411:411],T[0:7][0:41][203:203][411:411],time,lat[203:203],lon[411:411],lev","r")
					print("MERRA2_401.inst3_3d_asm_Np.20"+f"{l+1:02d}"+f"{m+1:02d}"+f"{n+1:02d}"+".nc4.ascii?PHIS[0:7][203:203][411:411],T[0:7][0:41][203:203][411:411],time,lat[203:203],lon[411:411],lev")
				else:
					f = open("MERRA2_400.inst3_3d_asm_Np.20"+f"{l+1:02d}"+f"{m+1:02d}"+f"{n+1:02d}"+".nc4.ascii?PHIS[0:7][203:203][411:411],T[0:7][0:41][203:203][411:411],time,lat[203:203],lon[411:411],lev","r")
					print("MERRA2_400.inst3_3d_asm_Np.20"+f"{l+1:02d}"+f"{m+1:02d}"+f"{n+1:02d}"+".nc4.ascii?PHIS[0:7][203:203][411:411],T[0:7][0:41][203:203][411:411],time,lat[203:203],lon[411:411],lev")
			#f = open("MERRA2_400.inst3_3d_asm_Np.2021"+f"{m+1:02d}"+f"{n+1:02d}"+".nc4.ascii?PHIS[0:7][203:203][411:411],T[0:7][0:41][203:203][411:411],time,lat[203:203],lon[411:411],lev","r")
			#print("MERRA2_400.inst3_3d_asm_Np.2021"+f"{m+1:02d}"+f"{n+1:02d}"+".nc4.ascii?PHIS[0:7][203:203][411:411],T[0:7][0:41][203:203][411:411],time,lat[203:203],lon[411:411],lev")
                
			lines = f.readlines()
			f.close()

			for i,line in enumerate(lines):
    				if i not in list:				# removing irrelevant rows 
        				data = line.split(",")
        				T.append(float(data[1]))		# storing temperature data

			T = np.array(T)

			#print(T)
		

			T1 = []

			count = 0
			sum1 = 0
			sum2 = 0
			index = 0
		
			for i in range(8):
    				for j in range(42):
        				if(8<j<31):
            					T1.append(T[count])
            					#print(count)
            					#print(index, x[index], dx[index], T[count])
            					sum1 = sum1 + np.exp(-x[index]/const)*T[count]*dx[index] 
            					#print(sum1)
            					sum2 = sum2 + np.exp(-x[index]/const)*dx[index] 
            					#print(sum2)
            					index = index+1
        				count = count+1
    				index = 0
    				effT.append(sum1/sum2)
    				sum1 = 0
    				sum2 = 0

			#print(T1)
			#print(len(T), len(T1))
			#print(effT)
			days = days + 1
	year_starts.append(24*days)
	
print(year_starts, year_names)

ax = plt.axes()
plt.minorticks_on()
ax.tick_params('both', which='major', length=6, width=1.1, direction='in', top=True, right=True)
ax.tick_params('both', which='minor', length=3, width=1, direction='in', top=True, right=True)
plt.plot(time, effT, 'r', linewidth=1)
plt.xlim(t_min, t_max)
plt.xticks(year_starts, year_names)
plt.xlabel("Time$\,$(year)", size='xx-large')
plt.ylabel(r'T$_\mathrm{eff}\,$(K)',size='xx-large')
#plt.legend()
plt.savefig("effTemp2006_10.pdf")
plt.show()
np.save("effTemp2001_20", effT)
np.savetxt("effTemp2005_10.txt",effT)
np.savetxt("Time2006_10.txt",time/24.0)

