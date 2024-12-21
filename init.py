from random import randint

batch_file = open('batch.dat', 'w')
batch_file.write(str(0) + ' ' + str(randint(0, 2)) + '\n')
batch_file.close()
