from multiprocessing import Pool, Queue
import os
import time
import uuid
import subprocess
import queue
from collections.abc import Iterable


def build_command(args, base_command):
    command = [""]
    for k in args:
        tmp = command
        command = []
        if not isinstance(args[k], Iterable):
            args[k] = [args[k]]
        for _v in args[k]:
            command.extend([x + " --" + k + "=" + str(_v) for x in tmp])
    command = [base_command + " " + x.strip() for x in command]
    command = ["export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && " + x for x in command]
    return command


def runCommand_xxx(gpu_idx, command, task_name):
    filename = str(uuid.uuid4().hex) + ".log"
    with open(os.path.join("log", task_name, filename), "w") as log:
        log.write(command + "\n")
        log.flush()
        run_command = command + (" --device=" + str(gpu[gpu_idx]))
        print("RUN COMMAND : " + run_command)
        res = subprocess.run(run_command, shell=True, stdout=log, stderr=log)
        if res.returncode == 0:
            log.write("\n" + filename + ":SUCCESS")
        else:
            task_queue.put(command)
        log.flush()
    print("END COMMAND : " + command)
    return gpu_idx


def release_gpu(idx):
    used[idx] -= 1


task_queue = None


def run_all(command, gpu, task_name, max_task_per_gpu=1):
    global used, task_queue
    used = [0] * len(gpu)
    if not os.path.isdir(os.path.join("log", task_name)):
        os.mkdir(os.path.join("log", task_name))

    for f in os.listdir(os.path.join("log", task_name)):
        if os.path.isfile(os.path.join("log", task_name, f)):
            lines = open(os.path.join("log", task_name, f), "r").readlines()
            if lines[-1].strip() == f + ":SUCCESS":
                if lines[0].strip() in command:
                    print("find success task, remove command :" + lines[0])
                    command.remove(lines[0].strip())
    task_queue = Queue()
    for c in command:
        task_queue.put(c)

    p = Pool(len(gpu)*max_task_per_gpu)
    while True:
        try:
            # empty_gpu_idx = -1
            min_used = min(used)
            if min_used < max_task_per_gpu:
                empty_gpu_idx = used.index(min(used))
                used[empty_gpu_idx] += 1
                p.apply_async(func=runCommand_xxx,
                              args=(empty_gpu_idx, task_queue.get(block=True, timeout=10), task_name),
                              callback=release_gpu)
            else:
                time.sleep(1)
                continue
            # for i in range(len(used)):
            #     if used[i] < max_task_per_gpu:
            #         empty_gpu_idx = i
            #         break
            # if empty_gpu_idx < 0:

            # empty_gpu_idx = used.index(0)

        except ValueError:
            time.sleep(10)
        except queue.Empty:
            p.join()
            if task_queue.empty():
                print('All subprocesses done.')
                break
    p.close()


if __name__ == '__main__':
    task_name = "cora_l2_test"
    args = {}
    # args = {
    #     "num_star": [1],
    #     "cross_star": [True],
    #     "dropout": [0.6],
    #     "hidden": [2048],
    #     "lr": [2e-4],
    #     "num_layers": [3, 6, 8],
    #     "cross_layer": [True, False],
    #     "relation_score_function": ["DistMult"],
    # }

    args["lr"] = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    args["cross_layer"] = [False, True]
    args["star_init_method"] = ["mean", "attn"]
    args["num_layers"] = [3, 6, 8]
    args["hidden"] = [128, 512, 1024, 2048]
    args["dropout"] = [0.2, 0.4, 0.6, 0.8]
    args["l2"] = [1.5e-4, 5e-4, 1.5e-3, 5e-3, 1.5e-4]
    args["additional_self_loop_relation_type"] = [False, True]

    gpu = [
        # 0,
        1,
        2,
        # 3,
        # 4,
        5,
        6,
        7,
    ]
    base_command = "/home/aarc/tianye/anaconda3/envs/pyg/bin/python -u /mnt/nas1/users/tianye/projects/graph_star/run_cora.py"
    command = build_command(args, base_command)
    run_all(command, gpu, task_name,max_task_per_gpu=10)
