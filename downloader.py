import hashlib
import os
import tarfile
import zipfile
import requests


DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir='./data'):
  assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}"
  url, sha1_has = DATA_HUB[name]
  os.makedirs(cache_dir, exist_ok=True)
  file_name = os.path.join(cache_dir, url.split('/')[-1])

  if os.path.exists(file_name):
    sha1 = hashlib.sha1()
    with open(file_name, 'rb') as f:
      while True:
        data = f.read(1048576) # why 1048576?
        if not data:
          break
        sha1.update(data)
    if sha1.hexdigest() == sha1_has:
      return file_name  # hit cache
    else:
      print(sha1.hexdigest())

  print(f'Downloading {url} to {file_name}...')
  r = requests.get(url, stream=True, verify=True)
  with open(file_name, 'wb') as f:
    f.write(r.content)
  return file_name

def download_extract(name, folder=None):
  file_name = download(name)
  base_dir = os.path.dirname(file_name) if folder is None else folder
  print(f'Extracting {file_name} to {base_dir}...')
  if file_name.endswith('.zip'):
    with zipfile.ZipFile(file_name, 'r') as z:
      z.extractall(base_dir)
  elif file_name.endswith(('.tar.gz', '.tgz')):
    with tarfile.open(file_name, 'r:gz') as t:
      t.extractall(base_dir)
  else:
    raise ValueError(f'Unsupport file type: {file_name}')
  return base_dir

def download_all():
  for name in DATA_HUB:
    download(name)