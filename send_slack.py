import argparse
import sys
from jhutil import send_slack

message = " ".join(sys.argv[1:])
send_slack(message)