


from math import isnan


class EvaluationEntry():
    def __init__(self,type,path,time,loss,epe,e3p):
        self.type = type
        self.path = path
        self.time = float(time)
        self.loss = float(loss)
        self.epe  = float(epe)
        self.e3p  = float(e3p)
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
    entries_dict = {e.path:e for e in entries}
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
        count+= 1
        entries_2.append(e)
    entries = entries_2
    print("LOSS",loss_avg/count)
    print("EPE ",epe_avg/count)
    print("E3P ",e3p_avg/count)
    print("MIN LOSS",min(entries,key=lambda x:x.loss).loss)
    print("MIN EPE ",min(entries,key=lambda x:x.epe).epe)
    print("MIN E3P ",min(entries,key=lambda x:x.e3p).e3p)

if __name__ == "__main__":
    log_folder = "/home/barny/Desktop/testlogs/" 
    ally = set()
    i=0
    check_log(f"/home/barny/Desktop/testlogs/sceneflow-testset-left_pad-1-3.log")
    for i in range(0):
        fname = f"/home/barny/Desktop/testlogs/sceneflow-testset-left_pad-{i}.log"
        print(fname)
        try:
            check_log(fname)
        except FileNotFoundError:
            pass