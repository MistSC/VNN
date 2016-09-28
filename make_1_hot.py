import numpy as np


for name in ['train','test','valid']:
    dataset=np.load(name+'.npy')
    l=[ 0 for i in range(10)]
    ll=[l[:] for i in range(dataset[0].shape[0])]
    print(dataset[0].shape[0])
    x,y=dataset

    for i in range(dataset[0].shape[0]):
        ll[i][y[i]]=1

    x=list(x)
    ll=list(ll)
    # do normalize
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j]=x[i][j]/255.

    dataset=x,ll
    dataset=np.array(dataset)
    np.save(name+'_1ok',dataset)




