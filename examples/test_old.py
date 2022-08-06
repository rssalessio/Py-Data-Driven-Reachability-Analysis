from zonotope import Zonotope
from matrix_zonotope import MatrixZonotope
import numpy as np

dim_x = 2

X0 = Zonotope(np.array(np.ones((dim_x, 1))), 0.1 * np.diag(np.ones((dim_x, 1)).T[0]))
U = Zonotope(np.array(np.ones((dim_x, 1))),0.25 * np.diag(np.ones((dim_x, 1)).T[0]))
W = Zonotope(np.array(np.zeros((dim_x, 1))), 0.003 * np.ones((dim_x, 1)))


total_samples = 100
"""index=1;
for i=1:size(W.generators,2)
    vec=W.Z(:,i+1);
    GW{index}= [ vec,zeros(dim_x,totalsamples-1)];
    for j=1:totalsamples-1
        GW{j+index}= [GW{index+j-1}(:,2:end) GW{index+j-1}(:,1)];
    end
    index = j+index+1;
end
"""

GW = []
for i in range(W.generators.shape[1]):
    vec = np.reshape(W.Z[:, i + 1], (dim_x, 1))
    dummy = []
    dummy.append(np.hstack((vec, np.zeros((dim_x, total_samples - 1)))))
    #print(vec.shape, W.Z.shape, dummy[i][:, 2:].shape, dummy[0][:, 0].shape)
    for j in range(1, total_samples, 1):
        right = np.reshape(dummy[i][:, 0:j], (dim_x, -1))
        left = dummy[i][:, j:]
        dummy.append(np.hstack((left, right)))
    GW.append(np.array(dummy))

GW = np.array(GW)
print(np.array(GW).shape)
print(W.generators.shape)