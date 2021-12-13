from lab_math import *

T1=40*(10**(-3)) #частота дискретизации
sig1=[]
sig2R=[]
sig2G=[]
sig2B=[]

f1=open("Contact.txt",'r')
f2=open("Video.txt",'r')

for line1 in f1:
    a=line1.split(',')
    sig1.append(float(a[1]))
f1.close()
for line1 in f2:
    a=line1.split(',')
    sig2R.append(float(a[1]))
    sig2G.append(float(a[2]))
    sig2B.append(float(a[3]))

f2.close()
sig1=np.array(sig1)/max(sig1)
sig2R=np.array(sig2R)/max(sig2R)
sig2G=np.array(sig2G)/max(sig2G)
sig2B=np.array(sig2B)/max(sig2B)

t=np.linspace(0, len(sig1), len(sig1))
t*=T1

s=sig1
s=butter_bandpass_filter(s,0.5,4,25)

f,A=spectrum(s,25)
fig, ax = plt.subplots(1,2)
ax[0].grid()
ax[0].plot(f[0:300],A[0:300])
ppg1,m1=ppg(f,A)
ax[0].plot(ppg1,m1,'*')
print('ЧСС 1')
print(ppg1*60)

B=gaussian_filter(A,3)
B=A*(B/max(B))
ax[1].grid()
ax[1].plot(f[0:300],B[0:300])
ppg2,m2=ppg(f,B)
ax[1].plot(ppg2,m2,'*')
print('ЧСС 2')
print(ppg2*60)

fig.savefig('Спектральные плотности', dpi = 1000)
print('SNR')
print(snr(A))

fi, a = plt.subplots()
a.plot(t[100:400],s[100:400])
a.plot(t[100:400],irfft(B)[100:400])

sig1=butter_bandpass_filter(sig1,0.5,4,25)
sig2G=butter_bandpass_filter(sig2G,0.5,4,25)

fi, a = plt.subplots()
a.grid()
a.plot(t[100:300],sig1[100:300])
a.plot(t[100:300],sig2G[100:300])
fi.savefig('Сигналы 1 -синий, сигнал зелёный - оранжевый', dpi = 1000)

print('r')
print(r(sig1,sig2G))

plt.show()