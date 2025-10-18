import requests


def query_freebase(entity_name):
    # FreeBase的查询API
    endpoint = "https://www.googleapis.com/freebase/v1/mqlread"
    query = {
        "query": {
            "name": "m.04w50v4",
            "type": "/people/person",  # 这里可以更改为其他类型，例如"/music/artist"等
        },
        "limit": 1
    }

    # 发送HTTP GET请求到FreeBase API
    response = requests.get(endpoint, params={"query": query})

    # 检查请求是否成功
    if response.status_code == 200:
        result = response.json()
        if result.get('result') is not None:
            return result['result']
        else:
            return None
    else:
        print("查询失败，状态码:", response.status_code)
        return None


# 使用示例
entity = query_freebase("Steve Jobs")
if entity:
    print(entity)
else:
    print("未找到相关实体")