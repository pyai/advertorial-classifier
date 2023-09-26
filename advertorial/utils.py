import os
import re
import logging
import os.path
from datetime import date

_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter('(%(module)s) [%(levelname)s] - %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_ch)


def check_env(envfile:str='.env'):
    """overwrite environment variables listed in `envfile`

    Args:
        envfile (str): environment variable file. Defaults to '.env'.
    """
    with open(envfile, 'r') as envfile:
        for line in envfile.readlines():
            line = line.replace('\n','')
            if line and '=' in line and '#' != line[0] and '=' !=line[-1]:
                key, value = line.split('=')
                print(f'env var: {key}={value}')
                os.environ[key]=value

def set_today(file = 'today.txt'):
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
