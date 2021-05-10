import os.path as osp
import os
import shutil

def copy_log(src, tgt):
    for fn in os.listdir(src):
        if "goal" in fn:
            src_fp = osp.join(src, fn, 'log.txt')
            tgt_fp = osp.join(tgt, fn, 'log.txt')
            shutil.copy(src_fp, tgt_fp)

if __name__ == '__main__':
    src = "/data1/PUBLIC/FOCAL-data/data_copy/walker_randparam_new"
    tgt = "/data1/PUBLIC/FOCAL-data/data_copy/walker_randparam_new_norm"
    copy_log(src, tgt)
