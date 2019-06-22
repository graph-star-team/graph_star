from tensorboardX import SummaryWriter
import os.path as osp
import time


train_steps = 0
val_steps = 0
epochs = 0
writer = None


def init_writer(name):
    global train_steps, val_steps, epochs, writer
    train_steps, val_steps, epochs = 0, 0, 0
    writer = SummaryWriter(osp.join("tensorboard", name+time.strftime("%y-%m-%d-%H-%M-%S")))

