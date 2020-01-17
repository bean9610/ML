import time
a1=time.time()
print(time.time())
a=time.strftime("%H:%M:%S")
print(time.strftime("%H:%M:%S")) ##24小时格式
print(time.strftime("%I:%M:%S"))## 12小时格式
time.sleep(2.5)
b1=time.time()
print(time.time())
b=time.strftime("%H:%M:%S")
print(time.strftime("%H:%M:%S")) ##24小时格式
print(time.strftime("%I:%M:%S"))## 12小时格式
print(time.strftime("%Y/%m/%d  %I:%M:%S"))## 带日期的12小时格式
print(b1-a1)
