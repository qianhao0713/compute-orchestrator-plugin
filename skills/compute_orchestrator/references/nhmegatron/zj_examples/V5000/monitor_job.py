# -*- coding: utf-8 -*-

import argparse
import csv
import datetime
import fcntl
import json
import logging
import os

# sys.path.append('/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages')
import re
import sys
import time
from pathlib import Path

import requests

csv.field_size_limit(sys.maxsize)
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 关键词： 训练
# 需要在megatron中打印训练结束标识符： pretrain结束时添加输出 print('[stop flag] train finnish')
# 断点保存模式：同步
# 基于megatron-baige分析

DATE_TYPE = '%Y-%m-%d_%H:%M:%S'  # 日期格式
logger = None


def get_args(local_test=False):
    required = not local_test
    parser = argparse.ArgumentParser(description="get args for monitor robot")
    parser.add_argument(
        "--local-test",
        type=bool,
        default=local_test,
        help="local_test. If set local_test, arguments will be set to the default value",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="/mnt/llama-test/qwen/qwen_zhijiang_megatron/"
        "Megatron_qwen_ls_1107_tz/log_tz/2024-12-17/2024-12-17_11:30_2n_monitor/0.log",
        required=required,
        help="log_path. Choose one log file to analyse",
    )
    parser.add_argument(
        "--webhook",
        type=str,
        default="https://oapi.dingtalk.com/robot/send?"
        "access_token=REPLACE_WITH_YOUR_DINGTALK_TOKEN",
        help="webhook. The webhook of dingding robot",
    )
    parser.add_argument(
        "--post-interval", type=int, default=500, help="post_interval. The interval to print log"
    )
    parser.add_argument(
        "--script",
        type=str,
        default="./train_qwen2.5_05b_mini.sh",
        required=required,
        help="script. Training script, use to distinguish messages",
    )
    parser.add_argument("--rank", type=int, default=0, required=required, help="rank. Node rank")
    parser.add_argument("--nnodes", type=int, default=1, required=required, help="nnodes. Node Num")
    parser.add_argument(
        "--job-name",
        type=str,
        default="ji-aitrain-8476268690702248877",
        required=required,
        help="JOB_NAME. for print_once",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen2.5",
        required=required,
        help="model_name. Use to distinguish messages",
    )
    parser.add_argument(
        "--pod-name",
        type=str,
        default="ji-aitrain-8488343569073627136-master-0",
        required=required,
        help="pod_name. Use to distinguish messages",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/mnt/llama-test/qwen/qwen_zhijiang_megatron/Megatron_qwen_ls_1107_tz/log_tz/",
        required=required,
        help="output_path. Path to save monitor log",
    )
    args = parser.parse_args()
    return args


def output_log(content_str, log_type='INFO'):
    global logger
    if log_type == 'INFO':
        logger.info(f'{content_str}')
    elif log_type == 'ERROR':
        logger.error(f'{content_str}')
    elif log_type == 'DEBUG':
        logger.debug(f'{content_str}')


def dingding_robot(webhook, content_str):
    data = {"msgtype": "text", "text": {"content": f'{content_str}'}}
    headers = {'Content-Type': 'application/json'}

    # r = requests.post(webhook, headers=headers, data=json.dumps(data))
    # r.encoding = 'utf-8'
    # errmsg = json.loads(r.text)['errmsg']
    # if errmsg != 'ok':
    #     ding_error = f'钉钉消息发送失败!!! {errmsg}'
    #     output_log(ding_error, log_type='ERROR')

    session = requests.Session()
    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
    )  # 配置重试和超时
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        r = session.post(webhook, headers=headers, data=json.dumps(data), timeout=20)  # 设置超时
        r.raise_for_status()  # 检查 HTTP 状态码
    except Exception as e:
        ding_error = f'钉钉消息发送失败!!! {e} \n 失败消息内容: {content_str}'
        output_log(ding_error, log_type='ERROR')


# log_type: 'INFO' , 'ERROR', 'DEBUG'
def print_content(args, content_ding, content_log, log_type='INFO'):
    if content_log:  # for log
        # print(f'-----------debug log----------- \n {content_log} \n ---------------------')
        output_log(content_log, log_type)

    if args.local_test:  # for local test
        print('content_ding:', content_ding)
        print('content_log:', content_log)
        return

    if content_ding:  # for robot
        # print(f'-----------debug ding----------- \n {content_ding} \n ---------------------')
        dingding_robot(args.webhook, content_ding)


# 只在rank 0 时写入
def print_content_0(args, content_ding, content_log, log_type='INFO'):
    if args.rank == 0:
        print_content(args, content_ding, content_log, log_type)


# 只在最后一个rank 写入
def print_content_last(args, content_ding, content_log, log_type='INFO'):
    if args.rank == args.nnodes - 1:
        print_content(args, content_ding, content_log, log_type)


# 只写入一次
def print_content_once(args, content_ding, content_log, str2find, log_type='INFO'):
    if not check_str_in_file(args.monitor_file_path, str2find):
        print_content(args, content_ding, content_log, log_type)


def check_str_in_file(file_path, str2find):
    return True if str2find in open(file_path, "r").read() else False


class KeyWords:
    def __init__(self):
        # self.training = 'throughput (tflops/sec/GPU)'
        self.training = 'throughput per GPU (TFLOP/s/GPU)'
        self.saved_ckpt = 'successfully saved checkpoint from iteration'
        # self.stop = '[stop flag] train finnish'
        self.stop = '[after training is done] datetime'


key_words = KeyWords()


class TimeRecord:
    def __init__(self):
        # ------------ 时间点 ------------
        self.train_start = 0  # 训练开始时间 (获取到日志的时间)
        self.load_ckpt_start = 0  # 开始加载断点的时刻
        self.load_ckpt_end = 0  # 加载断点结束的时刻
        self.saved_ckpt_last = 0  # 最近一次保存好断点的时刻
        self.first_iter = 0  # 第1个iter开始的时刻
        self.error = 0  # 读到Traceback的时刻
        # ------------ 时长 ------------
        self.init_model = 0  # 模型初始化时长 [开始有日志 --> 第1个iter前]
        self.load_ckpt = 0  # 加载断点耗时
        self.save_ckpt_last = 0  # 上一次保存断点耗时
        self.save_ckpt_total = 0  # 保存断点总耗时
        self.train_effective = 0  # 有效训练时长
        self.train_total = 0  # 总训练时长 [开始有日志 --> 监控结束]
        # 首次任务
        self.first_start = 0  # 首次任务启动的时刻
        # 上一次训练的信息
        self.last_job = self.TimeRecordLast()
        # 上一次故障的信息
        self.fault = self.TimeRecordFault()

        # 故障
        self.error_times = -1  # 故障次数
        self.run_times = -1  # 一次故障中的启动次数
        self.run_times_total = -1  # 总启动次数

    # 故障的时间信息
    class TimeRecordFault:
        def __init__(self):
            self.mttr = 0  # 故障恢复时长Mean Time to Recovery
            self.rolling_back = 0  # 一次故障中的，无效训练时长 [保存好断点 --> 故障前的最后1个iter] 或 (多次故障时)[第1个iter --> 故障前的最后1个iter]
            self.rolling_back_useful = 0  # 无效训练时长中的有效训练时长 (逐iter累加)
            self.mttd = 0  # 故障发现时长Mean Time to Detection [故障前的最后1个iter --> 重新开始的第1个iter之前]
            self.iter_start = 0  # 上一次训练时，第1个iter开始训练的时间
            self.iter_end = 0  # 上一次训练时，最后1个iter训完的时刻

    # 上一次训练的时间信息
    class TimeRecordLast:
        def __init__(self):
            # ------------ 时间点 ------------
            self.iter_effective = 0  # 上一次训练时，最后1个有效iter训完的时刻
            self.saved_ckpt = 0  # 上一次训练时，保存好最后一个断点的时刻

    # s转换为h
    def sec2hourstr(self, sec_in):
        minutes, sec = divmod(sec_in, 60)
        hours, minutes = divmod(minutes, 60)
        sec = round(sec, 0)  # 四舍五入
        return "%dh%dm%ds" % (hours, minutes, sec)

    # ['h', 'm', 's']转换为s
    def hourlist2sec(self, str):
        hour = int(str[0])
        minute = int(str[1])
        sec = int(str[2])
        sec = sec + minute * 60 + hour * 3600
        return sec

    # ['h', 'm', 's']转换为'xhxmxs'
    def hourlist2str(self, time_list):
        return f'{time_list[0]}h{time_list[1]}m{time_list[2]}s'

    # 从date str转换为时间戳
    def datestr2timestamp(self, str):
        return time.mktime(time.strptime(str, DATE_TYPE)) if str else 0

    # 从line中读取date str
    def line2datestr(self, line, idx=0):
        return re.findall(r'\d+\-\d+\-\d+\_\d+\:\d+\:\d+', line)[idx]

    # 从line中读取日期，并转换为时间戳
    def line2timestamp(self, line, idx=0):
        time_str = self.line2datestr(line, idx)
        return time.mktime(time.strptime(time_str, DATE_TYPE)) if str else 0

    # timestamp转为日期
    def timestamp2date(self, time_stamp):
        return datetime.datetime.fromtimestamp(time_stamp).strftime(DATE_TYPE)

    # 计算有效训练时间占比
    def count_effect_percent(self):
        percent_effective = self.train_effective / self.train_total * 100
        percent_effective = "{:.3f}".format(percent_effective)
        content_str = f'\n[有效训练时长] {self.sec2hourstr(self.train_effective)}'
        content_str += f'\n[总训练时长] {self.sec2hourstr(self.train_total)}'
        content_str += f'\n[有效训练时间占比] {percent_effective}%'
        return content_str


time_record = TimeRecord()


def set_log(args):
    global logger
    args.save_dir = os.path.join(args.output_path, 'monitor')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    args.pod_id = re.findall(r'\d+', args.pod_name)[0]
    args.monitor_file_path = os.path.join(args.save_dir, args.pod_id + '.log')

    logging.basicConfig(
        filename=args.monitor_file_path,
        format='%(asctime)s-%(levelname)s-%(process)d-%(message)s',
        datefmt=DATE_TYPE,
        level=logging.INFO,
    )
    logger = logging.getLogger()
    logger.handlers.clear()

    file_handler = logging.FileHandler(args.monitor_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s-%(levelname)s-%(process)d-%(message)s', DATE_TYPE)
    )
    logger.addHandler(file_handler)


# 获取文件的第一个时间戳
def get_first_time(file_path, reverse=False):
    global time_record
    ret = 0
    with open(file_path, 'r') as file:
        readlines = file.readlines()
    if reverse:
        readlines = reversed(readlines)
    for line in readlines:
        # 读第一个时间戳
        try:
            time_str = time_record.line2datestr(line)
            if line.startswith(time_str):
                ret = time_record.datestr2timestamp(time_str)
                break
        except:
            continue
    return ret


# 如果断点续训，需要解析上一份日志
def analyse_last_log(args, time_cur, iter_load):
    global time_record
    files = os.listdir(args.save_dir)
    files.sort()
    if len(files) < 2:  # 找不到其他日志
        print_content(args, None, '无可读监控日志')
        return

    files = files[:-1]  # 排除当前日志
    files = [os.path.join(args.save_dir, f) for f in files]  # 完整日志路径
    files_re = list(reversed(files))  # 按时间倒序排列日志

    # 有过训练的 file idx
    file_label = np.array([1 if check_str_in_file(f, key_words.training) else 0 for f in files_re])
    idx_train_file = np.where(file_label > 0)[0]

    # 从未训练过
    if idx_train_file.size == 0:
        # 故障恢复时长 = 故障发现时长 = 本次任务启动时刻 - 首次任务启动时刻
        time_record.fault.mttr = time_record.train_start - time_record.first_start
        time_record.fault.mttd = time_record.fault.mttr
        content_log = f'\n[故障恢复时长] {time_record.sec2hourstr(time_record.fault.mttr)}'
        content_log += f'\n[故障发现时长] {time_record.sec2hourstr(time_record.fault.mttd)}'
        print_content(args, None, content_log)
        return

    # 最多只分析2个日志：上一次训练的日志 + 上一次保存断点的日志
    idx_check_file = [idx_train_file[0]]
    file_label = np.array(
        [1 if check_str_in_file(f, f'{key_words.saved_ckpt} {iter_load}') else 0 for f in files_re]
    )
    idx_save_ckpt_file = np.where(file_label > 0)[0]
    if idx_save_ckpt_file.size > 0 and not idx_save_ckpt_file[0] in idx_check_file:
        idx_check_file.append(idx_save_ckpt_file[0])

    content_log = ''
    flag_ckpt_saved_time = False
    for idx in idx_check_file:
        file_path = files_re[idx]
        content_log += f'\n[加载监控日志] {file_path}'
        with open(file_path, 'r') as f:
            readlines = f.readlines()
        readlines_re = reversed(readlines)  # 倒着读日志
        for line in readlines_re:
            if key_words.training in line:
                iter = int(re.findall(r'\d+', line.split('consumed samples:')[0])[-2])
                iter_time = (
                    float(
                        re.findall(r'\d+\.\d+', line.split('elapsed time per iteration (ms):')[1])[
                            0
                        ]
                    )
                    / 1000
                )  # s
                time_record.fault.rolling_back_useful += iter_time
                # 上次故障，最后1轮训练结束的时刻
                if time_record.fault.iter_end == 0 and idx == idx_check_file[0]:
                    time_record.fault.iter_end = time_record.line2timestamp(line)
                    time_record.fault.mttd = time_cur - time_record.fault.iter_end
                    content_log += f'\n[最后1个step结束时刻] {time_record.timestamp2date(time_record.fault.iter_end)} at iter {iter}'
                    content_log += (
                        f'\n[故障发现时长] {time_record.sec2hourstr(time_record.fault.mttd)}'
                    )

                # 上次故障，第1轮训练开始的时刻
                if iter == iter_load + 1 and idx == idx_check_file[0]:
                    time_record.fault.iter_start = time_record.line2timestamp(line) - iter_time
                    time_record.fault.rolling_back = (
                        time_record.fault.iter_end - time_record.fault.iter_start
                    )
                    time_record.fault.mttr = time_record.fault.mttd + time_record.fault.rolling_back
                    content_log += f'\n[第1个step开始时刻] {time_record.timestamp2date(time_record.fault.iter_start)} at iter {iter}'
                    content_log += f'\n[回滚训练时长] {time_record.sec2hourstr(time_record.fault.rolling_back)}'
                    content_log += f'\n[回滚训练时长中的有效训练时长] {time_record.sec2hourstr(time_record.fault.rolling_back_useful)}'
                    content_log += (
                        f'\n[故障恢复时长] {time_record.sec2hourstr(time_record.fault.mttr)}'
                    )

                elif iter == iter_load:
                    time_str = time_record.line2datestr(line)
                    time_record.last_job.iter_effective = time_record.datestr2timestamp(time_str)
                    content_log += f'\n[上个有效训练时刻] {time_str} at iter {iter}'
                    break

            elif '[保存断点总时长]' in line and time_record.save_ckpt_total == 0:
                time_list = re.findall(r'\d+', line)[-3:]
                time_record.save_ckpt_total = time_record.hourlist2sec(time_list)
                content_log += f'\n[累计保存断点总时长] {time_record.hourlist2str(time_list)}'

            elif f'{key_words.saved_ckpt} {iter_load}' in line:
                flag_ckpt_saved_time = True
                time_str = time_record.line2datestr(line)
                time_record.last_job.saved_ckpt = time_record.datestr2timestamp(time_str)
                content_log += f'\n[最近一次保存断点的时刻] {time_str} at iter {iter_load}'

            elif '[有效训练时长]' in line and flag_ckpt_saved_time:
                time_list = re.findall(r'\d+', line)[-3:]
                time_record.train_effective = time_record.hourlist2sec(time_list)
                flag_ckpt_saved_time = False
                content_log += f'\n[累计有效训练时长] {time_record.hourlist2str(time_list)}'

    print_content(args, None, content_log)


def count_error_times(args):
    global time_record

    files = os.listdir(args.save_dir)
    files.sort()
    if len(files) < 2:  # 第一次监控
        time_record.error_times = 0
        time_record.run_times = 1
    else:
        files_re = list(reversed(files))
        files_re = files_re[1:]  # 排除当前日志

        error_list = [
            1 if check_str_in_file(os.path.join(args.save_dir, file), key_words.training) else 0
            for file in files_re
        ]
        time_record.error_times = sum(error_list)
        if time_record.error_times >= 1:
            time_record.run_times = np.where(np.array(error_list) == 1)[0][0] + 1
        else:
            time_record.run_times = len(error_list) + 1

        if error_list[-1] == 0:
            time_record.error_times += 1

    time_record.first_start = get_first_time(os.path.join(args.save_dir, files[0]))
    time_record.run_times_total = len(files)
    content_log = f'[首次任务启动时刻] {time_record.timestamp2date(time_record.first_start)}'

    content_log += f'\n[累计故障次数] {time_record.error_times}'
    content_log += f'\n[启动次数] {time_record.run_times}'
    content_log += f'\n[总启动次数] {time_record.run_times_total}'
    print_content_once(args, None, content_log, content_log)


def get_error_times_str(content_error_time):
    if content_error_time == '':
        with open(args.monitor_file_path, 'r') as file_tmp:
            readlines = file_tmp.readlines()
        for line_tmp in readlines:
            if '[累计故障次数]' in line_tmp:
                time_record.error_times = int(re.findall(r'\d+', line_tmp)[-1])
            if '[启动次数]' in line_tmp:
                time_record.run_times = int(re.findall(r'\d+', line_tmp)[-1])
            if '[总启动次数]' in line_tmp:
                time_record.run_times_total = int(re.findall(r'\d+', line_tmp)[-1])

        if time_record.error_times < 0:
            time_record.error_times = 1
        if time_record.run_times < 0:
            time_record.run_times = 1
        if time_record.error_times == 0 and time_record.run_times == 1:
            content_error_time = f'[故障计数] 训练任务的第1次启动'
        else:
            content_error_time = f'[故障计数] 第{time_record.error_times}次故障的第{time_record.run_times}次启动 (总启动次数{time_record.run_times_total})'
    return content_error_time


def init_events(args):
    events_save_path = os.path.join(args.output_path, 'events')
    if not os.path.exists(events_save_path):
        os.makedirs(events_save_path, exist_ok=True)
    args.events_file_path = os.path.join(events_save_path, 'events_record.csv')
    if not os.path.exists(args.events_file_path):  # 第一次创建时，写入表头
        with open(args.events_file_path, 'a', encoding='utf_8_sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['时间', '事件'])


# 将事件保存至csv
# 事件1: 任务启动
# 事件2: 当前任务的第1个step
# 事件3: Error信息
def save_events(args, events_str, time_stamp=None, flag_no_new_line=False, flag_once=False):
    file_path = args.events_file_path

    time_stamp = time.time() if time_stamp == None else time_stamp
    cur_time = time_record.timestamp2date(time_stamp)

    time_wait = 0
    while time_wait < 120:
        try:
            file = open(file_path, 'a+', newline='', encoding='utf_8_sig')
            fcntl.lockf(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # 加锁
            print_content(args, None, "acquire lock", 'DEBUG')
        except:
            print_content(args, None, "csv文件正在被其他进程使用", 'DEBUG')
            time.sleep(0.1)  # 等会儿
            time_wait += 0.1
            continue

        file.seek(0)
        data_list = list(csv.reader(file))  # 将文件内容转换成列表
        writer = csv.writer(file)

        # 只写入一次
        flag_write = True
        if flag_once and len(data_list) > 0:
            for rows in data_list:
                for row_col in rows:
                    if events_str in row_col:
                        flag_write = False
                        break
                if flag_write == False:
                    break

        # 第一次创建时，写入表头
        if len(data_list) == 0:
            writer.writerow(['时间', '事件'])
            print_content(args, None, "init csv", 'DEBUG')

        if flag_write:
            writer.writerow([f'{cur_time}', events_str])
            print_content(
                args, None, f"add line: [cur_time]{cur_time} - [events_str]{events_str}", 'DEBUG'
            )
        else:
            print_content(args, None, f"content exist: {events_str}", 'DEBUG')

        fcntl.lockf(file.fileno(), fcntl.LOCK_UN)  # 解锁
        print_content(args, None, "release lock", 'DEBUG')
        break

        # file.seek(0, 0)
        # data_list = list(csv.reader(file))  # 将文件内容转换成列表

        # # 只写入一次
        # flag_write = True
        # if flag_once and len(data_list) > 0 and events_str in data_list[-1][-1]:
        #     flag_write = False

        # # 第一次创建时，写入表头
        # if len(data_list) == 0:
        #     data_list.append(['时间', '事件'])
        #     print_content(args, None, "init csv", 'DEBUG')

        # if flag_write:
        #     if flag_no_new_line:
        #         data_list[-1][-1] += events_str # 修改最后一行
        #         print_content(args, None, f"add to last element: {events_str}", 'DEBUG')
        #     else:
        #         data_list.append([cur_time, events_str]) # 增加一行
        #         print_content(args, None, f"add line: [cur_time] {cur_time} - [events_str] {events_str}", 'DEBUG')
        #     # 写入
        #     file_write = open(file_path, 'w+', encoding='utf_8_sig')
        #     writer = csv.writer(file_write)
        #     writer.writerows(data_list)

        # fcntl.lockf(file.fileno(), fcntl.LOCK_UN) # 解锁
        # print_content(args, None, "release lock", 'DEBUG')
        # break


def run_monitor(args):
    global time_record
    line_num = 0  # 日志行数
    # node_info = "NULL" # 节点信息 # 无法获取节点信息
    iter_total = 0  # 总训练轮数
    iter_start = -1  # 第一个iter数，断点续训时不为1
    args.iter_cur = -1  # 当前iter数/读到的最后1个iter
    args.iter_saved = -1  # 保存断点的iter
    content_error_time = ''

    wait_log = 0
    while wait_log < 1200:
        if os.path.exists(args.log_path):
            break
        time.sleep(1)
        wait_log += 1
    if not os.path.exists(args.log_path):
        print(f'cannot find log {args.log_path} have been waiting for {wait_log} sec')
        return False

    # 初始化 events.csv
    events_save_path = os.path.join(args.output_path, 'events')
    if not os.path.exists(events_save_path):
        os.makedirs(events_save_path, exist_ok=True)
    args.events_file_path = os.path.join(events_save_path, 'events_record.csv')

    common_info = f'[监控日志] {args.monitor_file_path}'
    common_info += f'\n[训练脚本] {args.script}'
    common_info += f'\n[POD_NAME] {args.pod_name}'
    common_info += f'\n[JOB_NAME] {args.job_name}'

    # 监控开始，输出日志
    content_log = f'[{args.model_name}] 开始监控'
    content_log += f'\n[监控间隔] {args.post_interval}'
    content_log += f'\n[事件日志] {args.events_file_path}'
    content_log += f'\n{common_info}'
    print_content_once(args, None, content_log, '开始监控')

    with open(args.log_path, 'r') as f:
        # if not args.local_test:
        #     f.seek(0, 2) # Move the cursor to the end of the file
        post_cnt = 0
        while True:
            # Read new lines
            try:
                line = f.readline()
                line_num += 1
            except:  # No new lines, sleep for a while and retry
                time.sleep(0.1)
                continue
            line = line.strip()
            # 开始读取日志
            if line_num == 1:
                time_record.train_start = time.time()
                time_record.first_start = time_record.train_start

                # 读监控日志
                count_error_times(args)
                content_error_time = get_error_times_str(content_error_time)  # 统计故障次数
                print_content_once(args, None, content_error_time, content_error_time)

                content_events = f'任务启动'
                content_events += f'\n[POD_ID] {args.pod_id}'
                content_events += f'\n[JOB_NAME] {args.job_name}'
                content_events += f'\n{content_error_time}'
                content_log = f'记录任务启动信息到events.csv'
                save_events(args, content_events, flag_once=True)
                print_content_once(args, None, content_log, content_log)

                content_log = f'[{args.model_name}] 开始训练'
                content_log += f'\n[log] {args.log_path}'

                content_ding = f'[{args.model_name}] 开始训练'
                content_ding += f'\n{content_error_time}'
                content_ding += f'\n{common_info}'

                print_content_once(args, content_ding, content_log, '开始训练')

            # 获取总iter数
            if iter_total == 0 and "  train_iters ........." in line:
                iter_total = int(re.findall(r'\d+', line)[-1])
                content_log = f'[总训练轮数] {iter_total}'
                print_content_0(args, None, content_log)

            # elif 'save_interval .........' in line:
            #     args.save_interval = int(re.findall(r'\d+', line)[-1])
            #     content_log = f'[保存间隔] {args.save_interval}'
            #     print_content_0(args, None, content_log)

            # 判断是否保存断点
            # elif 'load ............................................' in line:
            #     print_content_0(args, None, f'{line}')

            # 获取节点信息
            # elif node_info == "NULL" and "infra-xpu-node-" in line:
            #     line_tmp = line.split('[')
            #     for info_tmp in line_tmp:
            #         if "infra-xpu-node-" in info_tmp:
            #             node_info = info_tmp.split(']')[0]
            #             break
            #     content_log = f'[rank{args.rank}节点信息] {node_info}'
            #     print_content(args, None, content_log)

            # 开始读取断点
            elif " loading " in line and " checkpoint " in line:
                time_record.load_ckpt_start = time.time()
            # 读取断点成功
            elif " successfully loaded checkpoint from" in line:
                time_record.load_ckpt_end = time.time()
                time_record.load_ckpt = time_record.load_ckpt_end - time_record.load_ckpt_start

                iter_load = int(re.findall(r'\d+', line)[-1])
                content_log = f'[{args.model_name}]'
                if iter_load > 0:
                    content_log += f' 成功加载断点 {line}'
                else:
                    content_log += f' 成功加载预训练模型 {line}'
                content_log += f'\n[加载耗时] {time_record.sec2hourstr(time_record.load_ckpt)}'
                content_ding = content_log + f'\n{common_info}'
                print_content_0(args, content_ding, content_log)

            # 保存断点标识
            # elif key_words.saved_ckpt in line:
            elif 'save-checkpoint' in line:
                # args.iter_saved = int(re.findall(r'\d+', line.split(key_words.saved_ckpt)[1])[0])
                args.iter_saved = args.iter_cur
                # cur_save_time = round(float(re.findall(r'\d+\.\d+', line.split('cost time')[1])[0]), 3)# s
                cur_save_time = float(re.findall(', *([\d\.]+)', line)[0]) / 1000.0

                if time_record.save_ckpt_total == 0 and check_str_in_file(
                    args.monitor_file_path, '[累计保存断点总时长]'
                ):
                    with open(args.monitor_file_path, 'r') as file_tmp:
                        readlines_tmp = file_tmp.readlines()
                    for line_tmp in readlines_tmp:
                        if '[累计保存断点总时长]' in line_tmp:
                            time_record.save_ckpt_total = time_record.hourlist2sec(
                                re.findall(r'\d+', line_tmp)[-3:]
                            )
                            break

                time_record.save_ckpt_last = cur_save_time
                time_record.save_ckpt_total += cur_save_time
                time_record.saved_ckpt_last = time.time()

                content_log = f'[保存断点] {line}'
                content_log += f'\n[单次保存断点时长] {cur_save_time}s'
                content_log += (
                    f'\n[保存断点总时长] {time_record.sec2hourstr(time_record.save_ckpt_total)}'
                )

                print_content_0(args, None, content_log)

                time_str = time_record.timestamp2date(time_record.saved_ckpt_last)
                content_log = f'[最近一次保存断点的时刻] at iter {args.iter_saved} 时间 {time_str}'
                print_content_0(args, None, content_log)

            # iter开始前的标识
            elif "[before the start of training step]" in line:
                time_record.init_model = time.time() - time_record.train_start
                content_log = f'[模型初始化时长] {time_record.sec2hourstr(time_record.init_model)}'
                print_content_0(args, None, content_log)

            # Check if the line contains "TFLOPS"
            elif key_words.training in line:
                # elif "throughput (token/sec/GPU) :" in line:
                time_cur = time.time()
                # 总iter数
                if iter_total == 0:
                    iter_total = int(re.findall(r'\d+', line.split('consumed samples:')[0])[-1])
                    content_log = f'[总训练轮数] {iter_total}'
                    print_content(args, None, content_log)

                # 统计有效训练时长（iter时长累加）
                args.iter_cur = int(re.findall(r'\d+', line.split('consumed samples:')[0])[-2])
                iter_time_cur = (
                    float(
                        re.findall(r'\d+\.\d+', line.split('elapsed time per iteration (ms):')[1])[
                            0
                        ]
                    )
                    / 1000
                )  # s 当前iter耗时
                if iter_start < 0:  # 第1个iter
                    iter_start = args.iter_cur

                    # 记录第1个iter开始的时间
                    time_record.first_iter = time_cur - iter_time_cur
                    save_events(
                        args, f'第1个step, at iter {iter_start}', time_stamp=time_record.first_iter
                    )

                    analyse_last_log(args, time_record.first_iter, args.iter_cur - 1)
                    print_content(
                        args,
                        None,
                        f'[本次训练开始时刻] {time_record.timestamp2date(time_record.first_iter)} at iter {iter_start}',
                    )

                # 计算有效训练时间占比
                time_record.train_effective += iter_time_cur
                time_record.train_total = time.time() - time_record.first_start
                content_time = time_record.count_effect_percent()
                print_content(args, None, line + content_time)

                if post_cnt % args.post_interval == 0 or args.iter_cur == iter_total:
                    content_error_time = get_error_times_str(content_error_time)  # 统计故障次数
                    content_ding = f'[{args.model_name}] 最新的训练日志 {line}'
                    content_ding += content_time
                    content_ding += f'\n{content_error_time}'
                    content_ding += f'\n{common_info}'
                    print_content(args, content_ding, None)

                post_cnt += 1

            # 运行报错
            elif "Traceback" in line:
                time_record.error = time.time()
                Error_str = ""
                Error_str_all = f"\n[详细报错信息] from log file {args.log_path} \n{line}\n"
                time_wait = 0
                while time_wait < 120:  # 最多等2分钟
                    # Read new lines
                    line_tmp = f.readline()
                    if not line_tmp:  # No new lines, sleep for a while and retry
                        time.sleep(0.1)
                        time_wait += 0.1
                        continue
                    Error_str_all += line_tmp
                    if "Error" in line_tmp:
                        Error_str = line_tmp
                        break

                if "ModuleNotFoundError: No module named 'flash_attn.layers'" in Error_str:
                    continue

                time_record.train_total = time.time() - time_record.first_start

                content_error_time = get_error_times_str(content_error_time)  # 统计故障次数

                # 记录任务异常信息到events.csv
                content_log = f'记录任务异常信息到events.csv'
                save_events(
                    args,
                    f'程序中断 \n[POD_ID] {args.pod_id} \n[JOB_NAME] {args.job_name}',
                    time_stamp=time_record.error,
                    flag_once=True,
                )
                print_content_once(args, None, content_log, content_log)
                save_events(args, f'{Error_str_all}', flag_no_new_line=True)

                content_ding = f'[{args.model_name}] 异常中断!'
                content_ding += f'\n{common_info}'
                # content_ding += f'\n[node] {node_info}'
                content_ding += f'\n[log] {args.log_path}'
                content_ding += f'\n[error] {Error_str.strip()}'
                content_ding += f'\n{content_error_time}'

                # 计算有效训练时间占比
                if args.rank == args.nnodes - 1:
                    if check_str_in_file(args.monitor_file_path, key_words.training):
                        content_time = time_record.count_effect_percent()
                        content_ding += content_time

                content_log = content_ding + Error_str_all
                print_content(args, content_ding, content_log, log_type='ERROR')
                return False

            # 运行结束
            elif "exiting program" in line or key_words.stop in line:
                # 记录任务完成信息到events.csv
                content_log = f'记录任务完成信息到events.csv'
                save_events(args, f'任务完成', flag_once=True)
                print_content_once(args, None, content_log, content_log)

                content_log = f'[{args.model_name}] 已完成，正常退出'

                content_error_time = get_error_times_str(content_error_time)  # 统计故障次数
                content_log += f'\n{content_error_time}'

                # 计算有效训练时间占比
                time_record.train_total = time.time() - time_record.first_start
                if check_str_in_file(args.monitor_file_path, key_words.training):
                    content_time = time_record.count_effect_percent()
                    content_log += content_time

                content_ding = content_log
                content_ding += f'\n{common_info}'
                print_content_last(args, content_ding, content_log)
                return True


if __name__ == "__main__":
    # for local test
    if len(sys.argv) < 2:
        flag_local_test = True
    else:
        flag_local_test = False

    args = get_args(flag_local_test)

    # 设置监控日志信息
    set_log(args)

    # 开启监控
    monitor_res = run_monitor(args)

    # 训练结束
    print_content(args, None, f'[训练结束] rank {args.rank}')

    if monitor_res == False:
        raise Exception("work fail!!")
