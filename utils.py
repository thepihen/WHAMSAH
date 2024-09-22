import sys
import numpy as np
import torch

def get_model(key, cfg):
    if key=='whamsah_net' or key.lower()=='whamsah' or key.lower()=='whamsahnet':
        from model.whamsah_net import WHAMSAHNet
        model = WHAMSAHNet(cfg)
    elif key=='whamsah_mix':
        from model.whamsah_net_mix import WHAMSAHNet
        model = WHAMSAHNet(cfg)
    elif key=='whamsah_mix_inv':
        from model.whamsah_net_mix_inv import WHAMSAHNet
        model = WHAMSAHNet(cfg)
    #ADD NEW MODELS HERE
    else:
        print(f"Invalid model: {key}. See utils.py for valid choices")
        sys.exit(1)
    return model

#Adapted from https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/utils.py
def load_weights(model, weights, verbose=False, useCuda=True):
    new_model = model.state_dict()
    old_model = torch.load(weights, map_location='cpu' if not torch.cuda.is_available() or useCuda==False else 'cuda:0')
    if 'state' in old_model:
        # Fix for htdemucs weights loading
        old_model = old_model['state']

    for el in new_model:
        if el in old_model:
            if verbose:
                print('Match found for {}!'.format(el))
            if new_model[el].shape == old_model[el].shape:
                new_model[el] = old_model[el]
            else:
                if len(new_model[el].shape) != len(old_model[el].shape):
                    if verbose:
                        print('Action: Different dimension! Too lazy to write the code... Skip it')
                else:
                    #the number of dimensions is the same, but the shape is different
                    if verbose:
                        print('Shape is different: {} != {}'.format(tuple(new_model[el].shape), tuple(old_model[el].shape)))
                    ln = len(new_model[el].shape)
                    max_shape = []
                    slices_old = []
                    slices_new = []
                    for i in range(ln):#for each dimension of the new model
                        #take the max length of that dimension between the two models
                        max_shape.append(max(new_model[el].shape[i], old_model[el].shape[i]))
                        slices_old.append(slice(0, old_model[el].shape[i])) #the original length in the model
                        slices_new.append(slice(0, new_model[el].shape[i]))
                    # print(max_shape)
                    # print(slices_old, slices_new)
                    slices_old = tuple(slices_old) #tuple of slices
                    slices_new = tuple(slices_new)
                    max_matrix = np.zeros(max_shape, dtype=np.float32)
                    for i in range(ln):
                        max_matrix[slices_old] = old_model[el].cpu().numpy()
                    max_matrix = torch.from_numpy(max_matrix)
                    new_model[el] = max_matrix[slices_new]
                    #probably even if there are more elements than weights, the model
                    #will still work by picking just the ones it needs, otherwise this function
                    #would return an error whenever the new model is smaller than the old one
        else:
            if verbose:
                print('Match not found for {}!'.format(el))
    model.load_state_dict(
        new_model
    )