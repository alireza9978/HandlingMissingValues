import pandas as pd
import requests

from src.preprocessing.load_dataset import root

if __name__ == '__main__':
    api_key = "api_key=738864c7e21a512f880681fd2e0c5f735f7a7c6b"
    country = "country=IE"

    urls = [
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2023".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2022".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2021".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2020".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2019".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2018".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2017".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2016".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2015".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2014".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2013".format(api_key, country),
        "https://calendarific.com/api/v2/holidays?{}&{}&year=2012".format(api_key, country),
    ]

    payload = {}
    headers = {
        'Cookie': '__cfduid=dba9870a9f40f66b06859cc86a6efe3551616847310; PHPSESSID=dfp3to6r0avgpr329ola36b4ju'
    }

    responses = []
    for url in urls:
        response = requests.request("GET", url, headers=headers, data=payload)
        responses.append(response.json())

    df = pd.DataFrame(pd.date_range(start='1/1/2012', end='1/1/2023').to_series(), columns=['date'])
    df['day_of_week'] = df.date.dt.dayofweek
    df['holiday'] = False
    df['weekend'] = False

    for res in responses:
        for holiday in res['response']['holidays']:
            date = holiday['date']['iso']
            df.loc[df['date'] == str(pd.to_datetime(date).date()), 'holiday'] = True

    df.loc[(df['day_of_week'] == 5) | (df['day_of_week'] == 6), 'weekend'] = True
    df.to_csv(root + "datasets/holiday.csv", index=False)
