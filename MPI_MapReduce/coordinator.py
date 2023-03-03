import logging
import os
from mpi4py import MPI
from model import Phase, Tag, MapMessage, ReduceMessage
from pathlib import Path
import time

from utils import create_path, group_files

logging.basicConfig(level=logging.INFO, filename='utils/map_reduce.log')

status = MPI.Status()



class Coordinator(object):
    def __init__(self, input_path: str, output_path: str, comm_world: MPI.Intracomm):
        self.__input_path = input_path
        self.__output_path = output_path
        self.__intermediate_path = "data/interm/"
        self.__comm_world = comm_world

        self.__input_files = []
        self.__currentPhase = Phase.PREPROCESSING
        self.__is_finished = False

    def run(self):
        while not self.__is_finished:
            match self.__currentPhase:
                case Phase.PREPROCESSING:
                    self.preprocessing()
                case Phase.MAP:
                    self.map()
                case Phase.INTERMEDIATE:
                    self.intermediate_processing()
                case Phase.REDUCE:
                    self.reduce()
                case Phase.FINISH:
                    self.finish()

    def preprocessing(self):
        logging.info("[COORD]: Preprocessing phase")
        for file in os.listdir(self.__input_path):
            file_path = Path(self.__input_path, file)
            self.__input_files.append(str(file_path))

        for worker_rank in range (0,self.__comm_world.size-1):
            worker_interm_path = self.__intermediate_path + str(worker_rank)
            worker_interm_path = Path(worker_interm_path)
            worker_interm_path.mkdir(parents=True,exist_ok=True)

        self.__input_files = group_files(files=self.__input_files,group_size=2)
        self.__currentPhase = Phase.MAP

    def map(self):
        # get the start time
        st = time.time()

        logging.info("[COORD]: Map phase")

        self.assign_tasks(output_path=self.__intermediate_path,tag=Tag.MAP_OPERATION.value,input_path=None)
        self.__currentPhase = Phase.INTERMEDIATE

        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = et - st

        logging.info('[COORD] Execution time for MAP is:{}'.format(elapsed_time))

    def intermediate_processing(self):
        logging.info("[COORD]: Intermediate processing phase")
        filenames = set()

        for (root, dirs, file) in os.walk(self.__intermediate_path):
            s = set(file)
            filenames.update(s)

        self.__input_files = list(filenames)
        self.__input_files.sort()
        self.__input_files = group_files(files=self.__input_files, group_size=50)
        self.__currentPhase = Phase.REDUCE

    def reduce(self):
        # get the start time
        st = time.time()

        logging.info("[COORD]: Reduce phase")

        self.assign_tasks(output_path=self.__output_path,tag=Tag.REDUCE_OPERATION.value,input_path=self.__intermediate_path)
        self.__currentPhase = Phase.FINISH

        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = et - st

        logging.info('[COORD] Execution time for REDUCE  is:{}'.format(elapsed_time))

    # anunt toti worker-ii ca trebuie sa isi incheie executia
    def finish(self):
        self.__is_finished = True

        for worker in range(0, self.__comm_world.size-1):
            logging.info("[COORD]: send FINISH to "+str(worker))
            self.__comm_world.send(0, dest=worker, tag=Tag.FINISH.value)


    def initialize_workers(self):
        workers = []
        for i in range(0, self.__comm_world.size - 1):
            workers.append(i)

        return workers

    def assign_tasks(self, output_path, tag, input_path):
        groups_number = len(self.__input_files)
        workers = self.initialize_workers()

        processed_groups = 0
        sent_groups_count = 0

        # pas 1 -> trimit la fiecare worker cate un grup de fisiere
        for current_worker in range(0, len(workers)):
            # daca sunt mai putine grupuri decat workeri
            if sent_groups_count == groups_number:
                break

            # trimit unui worker un fisier pentru procesare
            logging.info("[COORD]: send files {} - {} to {}".format(self.__input_files[sent_groups_count][0],
                                                                    self.__input_files[sent_groups_count][-1],
                                                                    current_worker))

            message = self.create_message(files_to_process=self.__input_files[sent_groups_count],
                                          output_path=create_path(tag, output_path, current_worker),
                                          input_path=input_path)
            self.__comm_world.send(message, dest=current_worker, tag=tag)
            sent_groups_count += 1

        # pas 2 -> astept procesarea task-urilor
        while processed_groups < groups_number:
            self.__comm_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.Get_source()
            response_tag = status.Get_tag()


            logging.info("[COORD]: recv " + Tag(response_tag).name + " from " + str(worker))
            processed_groups += 1

            # daca mai am task-uri, trimit unul worker-ului care tocmai a fost eliberat
            if sent_groups_count < groups_number:
                logging.info("[COORD]: send files {} - {} to {}".format(self.__input_files[sent_groups_count][0],
                                                                        self.__input_files[sent_groups_count][-1],
                                                                        worker))

                message = self.create_message(files_to_process=self.__input_files[sent_groups_count],
                                              output_path=create_path(tag, output_path, worker),
                                              input_path=input_path)
                self.__comm_world.send(message, dest=worker, tag=tag)
                sent_groups_count += 1

    def create_message(self, files_to_process, output_path,input_path):
        if input_path is None:
            return MapMessage(files_to_process=files_to_process, output_path=output_path)
        else:
            return ReduceMessage(input_path=input_path,files_to_process=files_to_process,output_path=output_path,max_dir_index=self.__comm_world.size-2)