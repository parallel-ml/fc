

def step(title, data):
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    for k in range(0, len(title), 68):
        print '+{:^68.68}+'.format(title[k:k+68])
    for k in range(0, len(data), 68):
        print '+{:^68.68}+'.format(data[k:k+68])
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print