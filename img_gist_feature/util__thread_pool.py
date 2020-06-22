# -*- coding: utf-8 -*-

import os
import sys
import queue
import threading

class UtilThreadPool():
    def __init__(self, n_thread_num=10, b_show=True, b_print=False):
        self.n_thread_num = n_thread_num
        self.n_todo_num = 0
        self.n_done_num = 0

        self.b_show = b_show
        self.b_print = b_print

        self.mutex = threading.Lock()
        self._run_logger = None
        
    
    def __multi_worker(self, func, qo_task_todo, qo_task_done):
        while not qo_task_todo.empty():
            o_task = qo_task_todo.get()
            n_ret, s_info = func(o_task)
            qo_task_done.put(o_task)

            if self.b_show:
                self.n_todo_num -= 1
                self.n_done_num += 1

                s_msg = 'Todo num:%d Done num:%d info:%s' % (self.n_todo_num, self.n_done_num, s_info)
                self._run_logger and self._run_logger.info(s_msg)
                self.b_print and print(s_msg)

    def main(self, o_info):
        self._run_logger = o_info["run_logger"]
        self.n_thread_num = o_info["n_thread_num"]
        self.b_show = o_info["b_show"] 

        func = o_info["func"]
        lo_task = o_info["lo_task"]

        qo_pic_todo = queue.Queue()  # 待处理数据队列
        qo_pic_done = queue.Queue()  # 已处理数据队列
        
        self.n_todo_num = len(lo_task)
        self.n_done_num = 0

        for o_task in lo_task:
            qo_pic_todo.put(o_task)    
            
        # 开启多线程处理队列的线程
        lt_thread = []
        for i in range(self.n_thread_num):
            t = threading.Thread(target=self.__multi_worker, args=(func, qo_pic_todo, qo_pic_done, ))
            t.setDaemon(True)
            t.start()
            lt_thread.append(t)

        for t in lt_thread:
            t.join()

        o_info["lo_task"] = []
        while not qo_pic_done.empty():
            o_info["lo_task"].append(qo_pic_done.get())


