class AverageMeter():
    
    def __init__(self):
        
        self.reset()

    def update(self, value, count):
        
        self.value = value
        self.sum += value * count
        self.count += count
        self.average = self.sum / self.count

    def reset(self):

        self.sum = 0
        self.average = 0
        self.count = 0
        self.value = 0