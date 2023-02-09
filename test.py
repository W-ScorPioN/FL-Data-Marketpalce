# import requests

# def getGlobalModel(taskid, round):
#     headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',}
#     url = 'http://10.114.100.154:9001/tee/getGlobalModel'
#     params = {"taskid":taskid, "round":round}
#     print(params)
#     response = requests.post(url=url, data=params, headers=headers).text
#     return response

# w_glob_str = getGlobalModel("123", "0")

print(type({"code":"200","message":"成功","result":"{'0': [0.088219708, 0.111540104, -0.283650290, 0.115413315, 0.269305184, -0.175225117, 0.117801669, -0.019594136, -0.085356664, 0.687946880]}"}))