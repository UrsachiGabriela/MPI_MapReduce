import os
import logging
import sys
from mpi4py import MPI
import time

from coordinator import Coordinator
from worker import Worker

logging.basicConfig(level=logging.INFO, filename='utils/map_reduce.log')

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == '__main__':

    if len(sys.argv) < 3:
        logging.error("Invalid args number")
        print("Invalid args number")
        sys.exit(1)

    if (not os.path.isdir(sys.argv[1])) or (not os.path.isdir(sys.argv[2])):
        logging.error("Invalid dir paths")
        print("Invalid dir paths")
        sys.exit(1)

    if rank == size - 1:
        # get the start time
        st = time.time()

        coord = Coordinator(input_path=sys.argv[1], output_path=sys.argv[2], comm_world=comm)
        coord.run()

        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = et - st

        logging.info('Execution time:{}'.format(elapsed_time))
    else:
        worker = Worker(comm_world=comm)
        worker.run()
