"""
ShiftLoss: 
    To compute the maximum pearson correlation between two time series within given lags.
    This allows misalignment between the two time series
"""

import torch
import torch.nn as nn

class ShiftLoss(nn.Module):
    '''first calcualte best shift by pearson correlation, then compute whatever loss'''
    def __init__(self, fn, shift=10):
        super().__init__()
        
        self.shift = shift
        self.fn = fn

        
    def forward(self, x, y):
        '''[batch, temporal]'''
        l = x.shape[0]
        resarr = []
        shift = self.shift
        
        for j in range(l):
            xs = x[j]
            ys = y[j]
            
            arr=[]
            for i in range(-shift, shift+1):
                xw,yw = self.getview(xs,ys, i)
                arr.append(torch.corrcoef(torch.stack([xw,yw],0)) [0][1])
            stacktens = torch.stack(arr,0)
            midx = torch.argmax(stacktens)
            
            xw,yw = self.getview(xs,ys, midx-shift)
            resarr.append(self.fn(xw.unsqueeze(0),yw.unsqueeze(0)))
        return torch.mean(torch.stack(resarr,0))
    
    def getview(self, xs, ys, sf):
        if sf < 0:
            xw = xs[-sf:]
            yw = ys[:sf]
        elif sf == 0:
            xw = xs
            yw = ys
        else:
            xw = xs[:-sf]
            yw = ys[sf:]
        
        return xw,yw