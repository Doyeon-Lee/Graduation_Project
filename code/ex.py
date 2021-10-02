import datetime
import time

x = 1730
hour = x // 25 // 60 // 60
minute = x // 25 // 60 % 60
second = x // 25 % 60
print(str(datetime.datetime.strptime(f'{hour}:{minute}:{second}', '%H:%M:%S').time()))
print(time.strftime('%H:%M:%S', time.gmtime(x // 25)))

