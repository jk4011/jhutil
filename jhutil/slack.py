import requests
import json
from .secret import SLACK_WEBHOOK_PROCESS, SLACK_WEBHOOK_JINHYEOK
from slack_sdk import webhook, WebClient
import random
from datetime import datetime
import jhutil


def send_slack(message, channel="gpu"):
    if channel == "gpu":
        webhook_url = SLACK_WEBHOOK_PROCESS
    elif channel == "jinhyeok":
        webhook_url = SLACK_WEBHOOK_JINHYEOK
    else:
        import jhutil
        jhutil.jhprint(1111, channel, "is not a valid channel name")
        raise NotImplementedError()
    payload = {
        "text": "<!channel>",
        "attachments": [{"text": message}]
    }
    headers = {
        "Content-type": "application/json; charset=utf-8"
    }
    response = requests.post(
        webhook_url, data=json.dumps(payload), headers=headers)
    # check response is valid
    if response.status_code != 200:
        # warning message
        print("Slack webhook failed with status code: {}".format(
            response.status_code))
        return False
    return True


def slack_wrapper(func, *args, **kwargs):
    print(0000, "started slack_wrapper")

    prcess_name = random.choice(prcess_name_set) + \
        "/" + random.choice(prcess_name_set)

    start = datetime.now()
    message = f"""START
prcess_name  : {prcess_name} 
time_start   : {start.strftime("%Y-%m-%d %H:%M:%S")}
    """
    jhutil.send_slack(message)

    try:
        # run function
        func(*args, **kwargs)
        end = datetime.now()

        message = f"""`END`
prcess_name  : {prcess_name} 
time_taken   : {end - start}
        """
        jhutil.send_slack(message)
        print(3333, message)

    except Exception as e:
        error_message = f"""*`Error`*!!!: 
```{str(e)}```
        """
        jhutil.send_slack(error_message)
        print(1111, error_message)


prcess_name_set = ["팬데믹",
                   "언택트",
                   "화양연화",
                   "공매도",
                   "윤달",
                   "치팅데이",
                   "비례_대표",
                   "거버넌스",
                   "간선상차",
                   "원더윅스",
                   "콘텐츠",
                   "인프라",
                   "주무관",
                   "코스피",
                   "버킷리스트",
                   "포스트_코로나",
                   "계획_관리_지역",
                   "레버리지",
                   "이데올로기",
                   "구상권",
                   "아방가르드",
                   "아카이브",
                   "6하원칙",
                   "코스닥",
                   "리즈시절",
                   "6.25전쟁",
                   "좁쌀_여드름",
                   "알콜성_치매",
                   "개인_파산",
                   "메타인지",
                   "핼러윈_데이", ]
