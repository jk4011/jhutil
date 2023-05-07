with open("./jhutil/secret.py") as f:
    data = f.read()

url_dict = {}

for line in data.strip().split("\n"):
    key, value = line.split(" = ")
    url_dict[key] = value.strip("\"")

# You can access the URLs using their keys as follows
SLACK_WEBHOOK_GPU = url_dict["SLACK_WEBHOOK_GPU"]
SLACK_WEBHOOK_JINHYEOK = url_dict["SLACK_WEBHOOK_JINHYEOK"]

import jhutil;jhutil.jhprint(1111, SLACK_WEBHOOK_GPU)
import jhutil;jhutil.jhprint(1111, SLACK_WEBHOOK_JINHYEOK)