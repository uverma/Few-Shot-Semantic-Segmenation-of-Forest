import numpy as np
import cv2
from parse import *
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import networkx as nx
from networkx.algorithms.flow import minimum_cut
import sys
from datetime import datetime
from matplotlib import image



class han:
    
    def __init__(self, flags, img, _mask, colors):
        
        self.FLAGS = flags
        self.ix = -1
        self.iy = -1
        self.img = img
        self.img2 = self.img.copy()
        self._mask = _mask
        self.COLORS = colors

    
    def h2(self, event, x, y, flags, param):

        # Draw the rectangle first
		self.FLAGS['DRAW_RECT'] = True
        self.ix, self.iy = x,y

        self.FLAGS['RECT'] = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
        self.FLAGS['rect_or_mask'] = 0

        # cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.COLORS['BLUE'], 2)
        self.FLAGS['RECT'] = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
        # self.FLAGS['rect_or_mask'] = 0

		cv2.circle(self.img, (x,y), 3, self.FLAGS['value']['color'], -1)
        cv2.circle(self._mask, (x,y), 3, self.FLAGS['value']['val'], -1)

        
def run(filename: str, numComp, gamma, it):

    
    COLORS = {
    'BLACK' : [0,0,0],
    'RED'   : [0, 0, 255],
    'GREEN' : [0, 255, 0],
    'BLUE'  : [255, 0, 0],
    'WHITE' : [255,255,255]
    }

    DRAW_BG = {'color' : COLORS['BLACK'], 'val' : 0}
    DRAW_FG = {'color' : COLORS['WHITE'], 'val' : 1}

    FLAGS = {
        'RECT' : (0, 0, 1, 1),     
        'DRAW_STROKE': False,      
        'DRAW_RECT' : False,       
        'rect_over' : False,       
        'rect_or_mask' : -1,       
        'value' : DRAW_FG,         
    }

    img = cv2.imread(filename)
#     img=cv2.resize(img,(int(0.3*img.shape[1]), int(0.3*img.shape[0])))
    img2 = img.copy()                                
    mask = np.zeros(img.shape[:2], dtype = np.uint8) 
    output = np.zeros(img.shape, np.uint8)           
    # cv2.namedWindow('1')
    # cv2.namedWindow('OP')
    
    EventObj = han(FLAGS, img, mask, COLORS)
    fn=filename.split('/')[-1].split('.')[0]

    while(1):
        
        img = EventObj.image
        mask = EventObj.mask
        FLAGS = EventObj.flags
        # cv2.namedWindow('1', cv2.WINDOW_GUI_NORMAL)
        # cv2.imshow('1', output)
        # cv2.namedWindow('2', cv2.WINDOW_GUI_NORMAL)
        # cv2.imshow('2', img)
        
        # k = cv2.waitKey(1)

            grabcut(img2.copy(), mask, FLAGS, numComp, gamma, it, fn)    
            EventObj.mask=mask
            
        EventObj.flags = FLAGS
        mask2 = np.where(((mask == 1)|(mask==2)), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img2, img2, mask = mask2)

def calculateBeta(img):
    diff_down=img[:-1,:]-img[1:,:]
    diff_right=img[:,:-1]-img[:,1:]
    beta_inv=2*np.sum(diff_down**2) + np.sum(diff_right**2)
    beta_inv/=((img.shape[1]-1)*img.shape[0]+(img.shape[0]-1)*img.shape[1])
    return 1/beta_inv
    

def findEdgeWeights(model, z, fgIndices, numComp):
    U=[]
    print(np.shape(fgIndices))
    for i in fgIndices:        
        comp=model.predict([z[i]])
        A=z[i]-model.means_[comp]
        D=-np.log(model.weights_[comp])+0.5*np.log(np.linalg.det(model.covariances_[comp]))+0.5*np.dot(A, np.dot(np.linalg.inv(model.covariances_[comp]), A.T))
        U.append(D)
# #     print(len(U))
    pre = np.zeros(numComp)
    inv = np.zeros((numComp,3,3))
    for comp in range(numComp):
      val = -np.log(model.weights_[comp])+0.5*np.log(np.linalg.det(model.covariances_[comp]))
      pre[comp]= val
      inval = np.linalg.inv(model.covariances_[comp])
      inv[comp] = inval

    C = model.predict(z[fgIndices])
    A = z[fgIndices] - model.means_[C]
    D1 = pre[C]
    

    A_ = np.reshape(A,(A.shape[0],A.shape[1],1))
    

    d2_ = np.matmul(inv[C], A_)
    

    A1 = np.reshape(A,(A.shape[0],1,A.shape[1]))


    D2 = np.matmul(A1,d2_).reshape(-1)
    D = D1 + 0.5 * D2
    U = list(D)
    return U
    

def constructGraph(img, bgModel, fgModel, mask, numComp, gamma):
    bgIndices=np.where(mask.reshape(-1)==0)[0]       #
    fgIndices=np.where(mask.reshape(-1)==1)[0]
    usIndices=np.where(mask.reshape(-1)==2)[0]


    beta=calculateBeta(img)
    
    source=img.shape[0]*img.shape[1]
    terminal=source+1

#     print(np.shape(usIndices))
    uBG=findEdgeWeights(bgModel, img.reshape(-1,3), usIndices, numComp)
    uFG=findEdgeWeights(fgModel, img.reshape(-1,3), usIndices, numComp)

    edges=[]
    edgeWeights=[]
    edges.extend(list(zip([source]*len(usIndices), usIndices)))
    edgeWeights.extend(uBG)

    edges.extend(list(zip([terminal]*len(usIndices), usIndices)))
    edgeWeights.extend(uFG)

    edges.extend(list(zip([source]*len(bgIndices), bgIndices)))
    edgeWeights.extend([0]*len(bgIndices))
#     print(len(edges)==len(edgeWeights))

    edges.extend(list(zip([terminal]*len(bgIndices), bgIndices)))
    edgeWeights.extend([sys.maxsize]*len(bgIndices))
#     print(len(edges)==len(edgeWeights))
    
    edges.extend(list(zip([source]*len(fgIndices), fgIndices)))
    edgeWeights.extend([sys.maxsize]*len(fgIndices))

    edges.extend(list(zip([terminal]*len(fgIndices), fgIndices)))
    edgeWeights.extend([0]*len(fgIndices))


    for col in range(img.shape[1]-1):
        for row in range(img.shape[0]):
            if(mask[row,col]!=mask[row,col+1]):
                if(mask[row,col]+mask[row,col+1]==1):
                    wt=0
                elif(mask[row,col]+mask[row,col+1]==2):
                    wt=gamma*np.exp(-beta*np.sum(np.square(img[row, col]-img[row,col+1])))
                else:
                    wt=2*gamma*np.exp(-beta*np.sum(np.square(img[row, col]-img[row,col+1])))
                edges.append((row*img.shape[1]+col, row*img.shape[1]+col+1))
                edgeWeights.append(wt)

    for row in range(img.shape[0]-1):
        for col in range(img.shape[1]):
            if(mask[row,col]!=mask[row+1,col]):
                if(mask[row,col]+mask[row+1,col]==1):
                    wt=0
                elif(mask[row,col]+mask[row+1,col]==2):
                    wt=gamma*np.exp(-beta*np.sum(np.square(img[row, col]-img[row+1,col])))
                else:
                    wt=2*gamma*np.exp(-beta*np.sum(np.square(img[row, col]-img[row+1,col])))
                edges.append((row*img.shape[1]+col, (row+1)*img.shape[1]+col))
                edgeWeights.append(wt)
#     print(len(edges)==len(edgeWeights))
    
    G = nx.Graph()
    G.add_nodes_from([i for i in range(terminal+1)])
    G.add_edges_from(edges)
    val={ edge: weight for (edge,weight) in zip(edges,edgeWeights)}
#     G.add_edges_from(val)
#     print(len(edges))
#     print(G.number_of_edges())
    nx.set_edge_attributes(G, val, 'capacity')
    cutValue, partition = minimum_cut(G, source, terminal, capacity='capacity')
#     print(cutValue)
    partition0=list(partition[0])
    partition1=list(partition[1])
    partition0.remove(source)
    partition1.remove(terminal)

    mask.reshape(-1)[np.where(mask.reshape(-1)[partition0]==0)]=1
    mask.reshape(-1)[partition1]=0
    

def grabcut(img, mask, FLAGS, numComp, gamma, iterations, fn):
    if(FLAGS['rect_or_mask']==0):
        startCol=FLAGS['RECT'][0]
        startRow=FLAGS['RECT'][1]
        endCol=FLAGS['RECT'][2]+startCol
        endRow=FLAGS['RECT'][3]+startRow  

     
        mask[startRow:endRow, startCol:endCol]=2       
        threshold=mask.shape[0]*mask.shape[1]*0.05 
        it=iterations
        FLAGS['rect_or_mask']=1
        maskCopy=mask.copy()
    else:
        threshold=mask.shape[0]*mask.shape[1]*0.01  
        maskCopy=mask.copy()
        it=3

    for i in range(it):
        bgIndices=np.where(mask==0)       
        fgIndices=np.where((mask==1)|(mask==2))

        bgModel=GaussianMixture(n_components=numComp, covariance_type="full", random_state=0).fit(img[bgIndices[0], bgIndices[1]])
        fgModel=GaussianMixture(n_components=numComp, covariance_type="full", random_state=0).fit(img[fgIndices[0], fgIndices[1]])

        constructGraph(img, bgModel, fgModel, mask, numComp, gamma)
        diff=np.sum(np.abs(maskCopy-mask))
#         print(diff)
        if(diff<threshold):
            break

    mask2 = np.where(((mask == 1)|(mask==2)), 255, 0).astype('uint8')
    output = cv2.bitwise_and(img, img, mask = mask2)
    now = datetime.now().strftime("%H:%M:%S") # time object
    outFname='../images/output/'+str(fn)+str(now)+'.jpg'
    print(outFname)
    image.imsave(outFname, cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
