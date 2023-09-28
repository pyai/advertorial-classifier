import os
import re
import logging
import os.path
from datetime import date
from typing import Optional

_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter('(%(module)s) [%(levelname)s] - %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_ch)


# def check_env(envfile:str='.env'):
#     """overwrite environment variables listed in `envfile`

#     Args:
#         envfile (str): environment variable file. Defaults to '.env'.
#     """
#     with open(envfile, 'r') as envfile:
#         for line in envfile.readlines():
#             line = line.replace('\n','')
#             if line and '=' in line and '#' != line[0] and '=' !=line[-1]:
#                 key, value = line.split('=')
#                 print(f'env var: {key}={value}')
#                 os.environ[key]=value

def check_env(envfile:str='.env'):
    """overwrite environment variables listed in `envfile`

    Args:
        envfile (str): environment variable file. Defaults to '.env'.
    """
     
    with open(envfile, 'r') as f:
         env_str = f.read()

    env_dict = {}
    # 使用正則表達式找到環境變數的設置行
    pattern = r'(\w+)=(.*)'
    matches = re.finditer(pattern, env_str)
    
    for match in matches:
        key = match.group(1)
        value = match.group(2)
        
        # 使用正則表達式查找並替換變數引用
        #pattern = r'\$(\w+)'
        pattern = r'\$\{?(\w+)\}?'
        value = re.sub(pattern, lambda m: env_dict.get(m.group(1), m.group(0)), value)
        
        # 將環境變數添加到字典中
        env_dict[key] = value
        print(f'env var: {key}={value}')
        os.environ[key] = value

def set_today(file = 'today.txt', today:str=''):
    pattern = r'20\d\d(0[1-9]|1[0-2])(0[1-9]|1[0-9]|2[0-9]|30|31)'
    result = re.match(pattern, today)
    
    if not result:
        today = date.today().strftime('%Y%m%d')

    open(file, 'w').write(today)
    return today

def get_today(file = 'today.txt'):
    today = open(file, 'r').read().replace('\n', '')
    pattern = r'20\d\d(0[1-9]|1[0-2])(0[1-9]|1[0-9]|2[0-9]|30|31)'
    result = re.match(pattern, today)
    if not result:
        today = set_today(file)

    return today

def get_based_path(path: str):
    # Access data from a mounted path of GCS which is built in the instance, like a VM for Vertex AI training job
    gcs_path = os.path.join("/gcs/milelens_ml/advertorial_post_classification", path)
    # Access data from local
    local_path = os.path.join("./", path)

    if os.path.exists(gcs_path):
        return gcs_path
    elif os.path.exists(local_path):
        return local_path
    else:
        raise FileNotFoundError(f"`{gcs_path}` or `{local_path}` not exists")
