"""
ShiftMSE: 
    To compute the minimum MSE between two time series within given lag range.
    This allows misalignment between the two time series
"""

import torch
import torch.nn as nn
# import pdb

class ShiftMSE(nn.Module):
    def __init__(self, shift=5) -> None:
        super().__init__()

        self.shift = shift
        self.fn = nn.MSELoss()

    def forward(self, x, y):
        '''[batch, temporal]'''
        # pdb.set_trace()
        l = x.shape[0]
        shift = self.shift
        resarr = []
        
        for j in range(l):
            xs = x[j]
            ys = y[j]
            
            arr=[]
            for i in range(-shift, shift+1):
                xw,yw = self.get_view(xs,ys, i)
                arr.append(self.fn(xw, yw))
                arr.append(torch.sqrt(torch.sum(torch.square(xw - yw))))
            stacktens = torch.stack(arr,0)
            resarr.append(torch.min(stacktens))
        return torch.mean(torch.stack(resarr,0))

    def get_view(self, xs, ys, sf):
        '''Return shifted array'''
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