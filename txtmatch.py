# -*- coding:utf-8 -*-
import re
with open('ceshi.txt','r',encoding='utf-8') as f1:
    line1=f1.read()
with open("wenben.txt",'r',encoding='utf-8') as f:
    line=f.read()
f2=open("biji.txt",'a+',encoding='utf-8')
f1.close()
f.close()
line=line.split("\n")
line1=line1.split("\n")
count=0
for i in line:
    if count <len(line1):
        # print(line1[count])
        # print(i.find(line1[count]))
        if i.find(line1[count])!=-1:
            if count==0:
                f2.write(line1[count] + "\n")
            else:
               f2.write("\n"+line1[count]+"\n")
            count+=1
            f2.write(i+",")
        else:
            f2.write(i+",")
    else:
        f2.write(i+",")
f2.close()


