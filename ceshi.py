with open('ceshi.txt','r',encoding='utf-8') as f1:
    line1=f1.readlines()
with open("wenben.txt",'r',encoding='utf-8') as f:
    line=f.readlines()
f1.close()
f.close()
for i in line:
    for j in line1:
        if j in i:
            print(i)
            print(j)