from mpi4py import MPI
import numpy as np
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--n', nargs='?', type=int, default=50,
                help='must be int')
arg = parser.parse_args()
n = arg.n



def prime_number(n):
    if n < 2 :
        return False
    for i in range(2,n):
        if n%i == 0:
            return False
    return True

def find_min_prime(number,rank):
    temp = number // 8 
    if 8 * temp + 2 * rank + 1 <= number :
        temp = temp + 1 
    while (1):
        check_number = temp * 8 + 2 * rank + 1
        # print(check_number)
        if prime_number(check_number):
            break
        temp += 1 
    return check_number


def find_twin_number(number, rank):
    while(1):
        n = find_min_prime(number, rank)
        if prime_number(n+2):
            return (n,n+2)
        else:
            number += 1


def count_twin(number, rank):
    bottom = int(rank)
    above = int(number)
    twin = []
    while(1):
        n = find_min_prime(bottom,rank)
        if n + 2 > above:
            break
        if prime_number(n+2):
            if (n,n+2) not in twin:
                twin.append ((n,n+2))
        bottom += 1
    return twin

            

if rank == 4:
    data = int(n) 
    for i in range(4):
        comm.isend(data, dest=i, tag=11)                           
        print("process {} immediate send {}...for problem 2".format(rank, data))
else:
#     for rank in range(4)
    data_count_twin =  count_twin(n,rank)
    data = comm.recv(source=4, tag=11)                         
    print("process {} recv {}...".format(rank, data)) 
    # data = data + 2 * rank +1 
    prime_number,twin_number = find_min_prime(data,rank),find_twin_number(data,rank)
    data = {'prime_number':prime_number,'twin_number':twin_number}
    comm.isend(data, dest=4, tag=12)
    print("process {} immediate send {}... for problem 4".format(rank, data))
  #  print('rank',rank, 'count', data_count_twin)
    comm.isend(data_count_twin, dest=4, tag=13)
    print("process {} immediate send {}... for problem 5".format(rank, data_count_twin))


data_twin = []
data_fin = []
if rank == 4:
    for i in range(4):
        temp = comm.recv(source = i, tag = 12)
        data_fin.append(temp)                           
        print("process {} immediate recv {}...".format(rank, temp))    


    least_min_twin = min([i['twin_number'][0] for i in data_fin])
    print('min_prime_number: ',min([i['prime_number'] for i in data_fin]), 'min_twin_number:  ',(least_min_twin,least_min_twin+2))

    for i in range(4):
        data_twin.extend(comm.recv(source = i, tag = 13))
    data_count = list(set(data_twin))
    print('the total number of twin numbers less than n: ',len(data_count))
    print(data_count)