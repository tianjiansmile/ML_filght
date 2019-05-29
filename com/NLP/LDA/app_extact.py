import pandas as pd
import json
import ast

def extact_app(x):
    if x:
        try:
            # d_info = eval(x)
            d_info = json.loads(x)
            # d_info = ast.literal_eval(x)
            # print(d_info)
            if d_info:
                device_info = d_info.get('device_info')
                if device_info:
                    appList = device_info.get('appList')
                    if appList and len(appList)>1:
                        return appList

                    applist = device_info.get('applist')
                    if applist and len(applist)>1:
                        # print(applist)
                        return applist

                    installedappsNew = device_info.get('installedappsNew')
                    a_list = []
                    if installedappsNew:
                        for app in installedappsNew:
                            app_name = app.get('app_name')
                            a_list.append(app_name)

                        return str(a_list)
                    # print(device_info)

                    return None
            return None
        except Exception as e:
            print('error',e,x)

    return None

if __name__ == '__main__':

    data = pd.read_csv('miaola_app.csv')

    data['app_list'] = data['device_info'].map(extact_app)

    data.dropna(axis=0, how='any', inplace=True)

    data.drop(['device_info'], axis=1, inplace=True)

    data.to_csv('miaola_extact.csv')

