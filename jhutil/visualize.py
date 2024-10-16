from IPython.display import HTML
from base64 import b64encode

def display_video(video_path):
  mp4 = open(video_path,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML("""
  <video width=600 controls>
    <source src="%s" type="video/mp4">
  </video>
  """ % data_url)