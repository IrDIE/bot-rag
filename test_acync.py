
import json

test_query_user = {
  "query": "В каком рейтинге (по состоянию на 2021 год) ИТМО впервые вошёл в топ-400 мировых университетов?\n1. ARWU (Shanghai Ranking)\n2. Times Higher Education (THE) World University Rankings\n3. QS World University Rankings\n4. U.S. News & World Report Best Global Universities",
  "id": 3
}
import  time
import aiohttp
import asyncio

URL = "http://localhost:8080/api/request"

question_wuth_answers = "В каком рейтинге (по состоянию на 2021 год) ИТМО впервые вошёл в топ-400 мировых университетов?\n1. ARWU (Shanghai Ranking)\n2. Times Higher Education (THE) World University Rankings\n3. QS World University Rankings\n4. U.S. News & World Report Best Global Universities"
async def send_request(session, data):
    async with session.post(URL, json=data) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, {
                "id": 0,
                "query": question_wuth_answers
                }),
            # send_request(session, {
            #     "id": 1,
            #     "query": "В каком городе находится ИТМО?"
            # }),
            # send_request(session, {
            #     "id": 2,
            #     "query": "как работает AI-шлем, разработанный стартапом сотрудников ИТМО?\n1. решает проблему подготовки космонавтов\n2. решает проблему искуственного интеллекта а главный принцип работы это VR очки\n3. шлем похож на канадскую разработку RealWear. Ключевым элементом шлема является прозрачный дисплей дополненной реальности в виде очков."
            # }),

            # send_request(session, {
            #     "id": 3,
            #     "query": question_wuth_answers
            # })
        ]
        responses = await asyncio.gather(*tasks)
        # print(responses)
        for resp in responses:
            print(resp)


start_time = time.time()
asyncio.run(main())
print(time.time()- start_time)


# if __name__=="__main__":
    
#     y = json.dumps(test_query_user)
#     print(y)