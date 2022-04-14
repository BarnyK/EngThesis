from asyncore import read
from math import isnan


class EvaluationEntry:
    def __init__(self, type, path, time, loss, epe, e3p):
        self.type = type
        self.path = path
        self.time = float(time)
        self.loss = float(loss)
        self.epe = float(epe)
        self.e3p = float(e3p)

    def __repr__(self):
        return f"{self.type},{self.time},{self.loss},{self.epe},{self.e3p}, {self.path}"

    def __str__(self):
        return self.__repr__()


def check_log(filename):
    with open(filename) as f:
        data = f.read().split("\n")
    data = [x for x in data if x]
    data = [x.split(",") for x in data]
    entries = [EvaluationEntry(*x) for x in data]
    entries_dict = {e.path: e for e in entries}
    loss_avg = 0.0
    epe_avg = 0.0
    e3p_avg = 0.0
    count = 0
    entries_2 = []
    while entries:
        e = entries.pop()
        if isnan(e.loss):
            continue
        loss_avg += e.loss
        epe_avg += e.epe
        e3p_avg += e.e3p
        count += 1
        entries_2.append(e)
    entries = entries_2
    print("LOSS", loss_avg / count)
    print("EPE ", epe_avg / count)
    print("E3P ", e3p_avg / count)
    print("MIN LOSS", min(entries, key=lambda x: x.loss).loss)
    print("MIN EPE ", min(entries, key=lambda x: x.epe).epe)
    print("MIN E3P ", min(entries, key=lambda x: x.e3p).e3p)


# if __name__ == "__main__":
#     log_folder = "/home/barny/Desktop/testlogs/"
#     ally = set()
#     i = 0
#     check_log(f"/home/barny/Desktop/testlogs/sceneflow-testset-left_pad-1-3.log")
#     for i in range(0):
#         fname = f"/home/barny/Desktop/testlogs/sceneflow-testset-left_pad-{i}.log"
#         print(fname)
#         try:
#             check_log(fname)
#         except FileNotFoundError:
#             pass

class LogEntry:
    def __init__(self, save,dataset, lr,epoch,type,iters, time, loss, epe, e3p,test_time,test_loss,test_epe,test_e3p):
        self.save = save
        self.dataset = dataset
        self.lr = float(lr)
        self.epoch = int(epoch)
        self.type = type
        self.iters = int(iters)
        self.time = float(time)
        self.loss = float(loss)
        self.epe = float(epe)
        self.e3p = float(e3p)
        self.test_time = float(test_time)
        self.test_loss = float(test_loss)
        self.test_epe = float(test_epe)
        self.test_e3p = float(test_e3p)

    def __repr__(self):
        return f"{self.type},{self.time},{self.loss},{self.epe},{self.e3p}"

    def __str__(self):
        return self.__repr__()
    @property
    def isEpoch(self):
        return self.type == "epoch"
    

    
def main4():
    from matplotlib import pyplot as plt
    import numpy as np
    filename = "/home/barny/data/logs/nosdea-sceneflow-1-1.log"
    with open(filename) as f:
        data = f.read().split("\n")
        data = [x for x in data if x]
        data = [x.split(",") for x in data[200:]]
        data = [LogEntry(*x) for x in data]
    epochs = [d for d in data if d.isEpoch]
    iters = [d for d in data if not d.isEpoch]
    filter = iters[-200:]
    losses = np.array([x.loss for x in filter])
    epes = np.array([x.epe for x in filter])
    e3ps = np.array([x.e3p for x in filter])

    m,b = np.polyfit(np.array([i for i in range(losses)]),losses,1)
    print(m,b)
    print(sum(losses)/len(filter))
    print(sum(epes)/len(filter))
    print(sum(e3ps)/len(filter))
    plt.figure(1)
    plt.plot([x.loss for x in iters])
    plt.title("Loss")
    plt.figure(2)
    plt.plot([x.epe for x in iters])
    plt.title("Endpoint error")
    plt.figure(3)
    plt.plot([x.e3p for x in iters])
    plt.title("3 pixel error")
    plt.show()


main4()

