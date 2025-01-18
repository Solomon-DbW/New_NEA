import datetime

todaydate = datetime.date.today()
todaydate = todaydate.strftime("%d-%m-%Y")
todaydate = todaydate.split("-")
print(todaydate)

# print(datetime.datetime.now())
