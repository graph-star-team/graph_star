from tensorboardX import SummaryWriter
import os.path as osp
import time


steps = 0
epochs = 0
writer = None

def log_epoch(DATASET, epoch, train_lp_auc, train_lp_ap, train_loss, val_lp_auc, val_lp_ap, val_loss, test_lp_auc, test_lp_ap, test_loss,max_lp_auc,max_lp_ap,max_val_lp):
    global steps, epochs, writer
    writer.add_scalar('train/lp_auc', train_lp_auc, steps)
    writer.add_scalar('train/lp_ap', train_lp_ap, steps)
    writer.add_scalar('train/loss', train_loss, steps)

    writer.add_scalar('val/lp_auc', val_lp_auc, steps)
    writer.add_scalar('val/lp_ap', val_lp_ap, steps)
    writer.add_scalar('val/loss', val_loss, steps)

    writer.add_scalar('test/lp_auc', test_lp_auc, steps)
    writer.add_scalar('test/lp_ap', test_lp_ap, steps)
    writer.add_scalar('test/loss', test_loss, steps)

    train_str = "LP AVG: {:.4f}, ".format(sum([train_lp_auc, train_lp_ap]) / 2)
    val_str = "LP AVG: {:.4f}, ".format(sum([val_lp_auc, val_lp_ap]) / 2)
    test_str = "LP AVG: {:.4f}, ".format(sum([test_lp_auc, test_lp_ap]) / 2)
    max_str = "LP AVG: {:.4f},VAL: {:.4f} ".format(sum([max_lp_auc, max_lp_ap]) / 2, max_val_lp)

    log_str = 'Epoch: {:02d}, TRAIN Loss: {:.4f}, {} || VAL Loss: {:.4f}, {} || TEST Loss: {:.4f}, {} || Max {}'.format(
        epoch, train_loss, train_str, val_loss, val_str, test_loss, test_str, max_str)
    print("\033[1;32m" + DATASET + " results:", "\033[0m" + '\n'+log_str)


def init_writer(name, info):
    global steps, epochs, writer
    steps, epochs = 0, 0
    writer = SummaryWriter(osp.join("tensorboard", name+'_'+time.strftime("%d-%m-%y %H-%M")))
    writer.add_text('model/info', info, 0)

