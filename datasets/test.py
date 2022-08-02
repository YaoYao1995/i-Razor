from __future__ import print_function

import time

from .Criteo_Challenge import Criteo_Challenge

dataset = Criteo_Challenge(False)
exit(0)



from Criteo_all import Criteo_all

d = Criteo_all(True, 9)

test_data_param = {
    'gen_type': 'test',
    'random_sample': False,
    'batch_size': 20000,
}

test_gen = d.batch_generator(test_data_param)

for i in range(3):
    flag = True
    for x in test_gen:
        if flag:
            print(x)
            flag = False
    print(i)

test_iter = iter(test_gen)
for i in range(3):
    try:
        flag = True
        while True:
            x = test_iter.next()
            if flag:
                print(x)
                flag = False
    except StopIteration, e:
        test_iter = iter(test_gen)
    print(i)
