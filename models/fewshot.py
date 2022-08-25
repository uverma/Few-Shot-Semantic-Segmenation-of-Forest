"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from .vgg import Encoder
from backbone.backbone import backbone
ci=3
class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    global ci
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)),]))


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        # print(fore_mask)
        # fg = fore_mask.cpu.numpy()
        # cv2.imshow("supp_img", fg)
        # cv2.waitKey(1000)
        backbone_model = backbone(ci)
        feature_maps = backbone_model(input_images)
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H x W
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H x W
        # print("foremask",fore_mask.shape)
        # print('backmask', back_mask.shape)
        # fore_mask=fore_mask.mean(5)
        # back_mask=back_mask.mean(5)
        #feature maps
        Channel_feature, Height_feature, Width_feature = feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3]
        ta_model = TANet(f_in=Channel_feature, C_in=Channel_input, kernel_sizes=[3, 5])
        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(batch_size):
            ###### Extract prototype ######

            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            
            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)
            # print("hello")
            # print(qry_fts.shape)
            # print(prototype.shape)
            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        # print(fts.shape)
        # print(prototype.shape)
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler

        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        # print("fts")
        # print(fts.shape)
        # print('mask')
        # print(mask.shape)
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts

    
    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding features for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss


#working
# fts torch.Size([1, 512, 16, 16])
# mask torch.Size([1, 128, 128])
# foremask torch.Size([1, 1, 1, 128, 128])
# backmask torch.Size([1, 1, 1, 128, 128])

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
            

        def gc(img, mask, FLAGS, numComp, gamma, iterations, fn):
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





def norm(ar):
    return 255.*np.absolute(ar)/np.max(ar)

I = cv2.imread('pebbles.jpg')
I2 = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
gray2 = np.copy(gray.astype(np.float64))
(rows, cols) = gray.shape[:2]

conv_maps = np.zeros((rows, cols,16),np.float64)

filter_vectors = np.array([[1, 4, 6,  4, 1],
                            [-1, -2, 0, 2, 1],
                            [-1, 0, 2, 0, 1],
                            [1, -4, 6, -4, 1]])


model=backbone(Channel)
op=model(ip)
op_f=model(ip)
filters = list()
for ii in range(4):
    for jj in range(4):
        feature_maps_update, att_map = ta_model(feature_maps, input_images)
        filters.append(np.matmul(filter_vectors[ii][:].reshape(5,1),filter_vectors[jj][:].reshape(1,5)))

smooth_kernel = (1/25)*np.ones((5,5))
gray_smooth = sg.convolve(gray2 ,smooth_kernel,"same")
gray_processed = np.abs(gray2 - gray_smooth)

for ii in range(len(filters)):
    conv_maps[:, :, ii] = sg.convolve(gray_processed,filters[ii],'same')

texture_maps = list()
texture_maps.append(norm((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2))
texture_maps.append(norm((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2))
texture_maps.append(norm(conv_maps[:, :, 10]))
texture_maps.append(norm(conv_maps[:, :, 15]))
texture_maps.append(norm((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2))
texture_maps.append(norm(conv_maps[:, :, 5]))
texture_maps.append(norm((conv_maps[:, :, 7]+conv_maps[:, :, 13])//2))
texture_maps.append(norm((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2))



# not working
# fts torch.Size([1, 512, 16, 16])
# mask torch.Size([1, 128, 128, 3])
# foremask torch.Size([1, 1, 1, 128, 128, 3])
# backmask torch.Size([1, 1, 1, 128, 128, 3])



