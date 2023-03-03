import os
from pathlib import Path

from mpi4py import MPI
from utils import remove_unnecessary_words
from model import Tag, MapMessage, ReduceMessage
import logging

logging.basicConfig(level=logging.INFO, filename='utils/map_reduce.log')
status = MPI.Status()

class Worker(object):

    def __init__(self, comm_world: MPI.Intracomm):
        self.__comm_world = comm_world
        self.__is_finished = False
        self.__coord = comm_world.size - 1
        self.__rank = comm_world.rank

    def run(self):
        while not self.__is_finished:
            data = self.__comm_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            source = status.Get_source()

            #print("[WORKER]" + str(self.__rank) + ": recv tag " + str(tag) + " with " + str(data) + " from " + str(source))

            match tag:
                case Tag.FINISH.value:
                    logging.info("[WORKER {}]: recv tag {} from {}".format(self.__rank, Tag(tag).name,
                                                                                        source))
                    self.finish_exec()
                case Tag.MAP_OPERATION.value:
                    logging.info("[WORKER {}]: recv tag {} with {} - {} from {}".format(self.__rank, Tag(tag).name,
                                                                                        data.files_to_process[0],
                                                                                        data.files_to_process[-1],
                                                                                        source))
                    self.map(data)
                case Tag.REDUCE_OPERATION.value:
                    logging.info("[WORKER {}]: recv tag {} with {} - {} from {}".format(self.__rank, Tag(tag).name,
                                                                                        data.files_to_process[0],
                                                                                        data.files_to_process[-1],
                                                                                        source))
                    self.reduce(data)

    def map(self, data:MapMessage):
        # procesare data
        files_to_process = data.files_to_process
        output_path = data.output_path
        words = {}

        for file_path in files_to_process:
            f = open(file_path, "r", errors='ignore')
            os.set_blocking(f.fileno(), False)
            lines = f.readlines()

            filename = os.path.basename(file_path)

            for line in lines:
                filtered_line = remove_unnecessary_words(line)
                for word in filtered_line:
                    words.setdefault(word, dict())

                    if words[word].keys().__contains__(filename):
                        words[word][filename] += 1
                    else:
                        words[word][filename] = 1

            f.close()

        for word in words:
            output_file = output_path + word + ".txt"
            f = open(output_file, 'a')

            for doc in words[word]:
                to_write = doc + ':' + str(words[word][doc])
                f.write(to_write + "\n")

            f.close()


        # trimit raspuns catre master sa il anunt ca am terminat
        logging.info("[WORKER {}]: send tag {} to {}".format(self.__rank,Tag.FREE_WORKER.name,self.__coord))
        self.__comm_world.send(0, dest=self.__coord, tag=Tag.FREE_WORKER.value)


    def reduce(self, data:ReduceMessage):
        # procesare data
        files_to_process = data.files_to_process
        input_path = data.input_path
        output_path = data.output_path
        max_dir_index = data.max_dir_index

        for word in files_to_process:
            my_dict = {}
            for i in range(0, max_dir_index + 1):
                file_path = input_path + str(i) + "/" + word

                if Path(file_path).is_file():
                    f = open(file_path, "r", errors='ignore')
                    os.set_blocking(f.fileno(), False)
                    lines = f.readlines()

                    for line in lines:
                        doc_id, count = line.split(':')

                        if my_dict.keys().__contains__(doc_id):
                            my_dict[doc_id] += int(count)
                        else:
                            my_dict[doc_id] = int(count)

                    f.close()

            f = open(output_path, 'a')
            word = word.split(".")[0]
            f.write(word + " ->  " + str(my_dict) + "\n")

            f.close()

        # trimit raspuns catre master sa il anunt ca am terminat
        logging.info("[WORKER {}]: send tag {} to {}".format(self.__rank,Tag.FREE_WORKER.name,self.__coord))
        self.__comm_world.send(0, dest=self.__coord, tag=Tag.FREE_WORKER.value)

    def finish_exec(self):
        self.__is_finished = True
        logging.info("[WORKER {}]: finish exec".format(self.__rank))