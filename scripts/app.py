import fire

from advertorial.train import train
from advertorial.performance import summary



def train_then_summary(envfile:str='.env', 
                       use_wandb:bool=True,):
    train(envfile=envfile, use_wandb=use_wandb)
    summary(envfile=envfile)

if __name__ == '__main__':
    fire.Fire({
        "train_then_summary": train_then_summary,
    })
