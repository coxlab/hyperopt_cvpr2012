import Image
import scipy as sp
import scipy.signal
import numpy as np

from npclockit import clockit_onprofile
import colorconv

from npclockit import clockit_onprofile

conv = scipy.signal.convolve

PROFILE = False

def preprocess(arr,config):
    if config.get('preproc'):
        orig_imga,orig_imga_conv = image_preprocessing(arr,config) 
    else:
        orig_imga = arr
        orig_imga_conv = orig_imga.copy()
    output = {}
    for cidx in xrange(orig_imga_conv.shape[2]): 
        if config.get('preproc'):
            output[cidx] = map_preprocessing(orig_imga_conv[:,:,cidx],config) 
        else:
            output[cidx] = orig_imga_conv[:,:,cidx]

    return output,orig_imga


# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def get_image(img_fname, max_edge=None, min_edge=None,
              resize_method='bicubic'):
    """ Return a resized image as a numpy array

    Inputs:
      img_fname -- image filename
      max_edge -- maximum edge length (None = no resize)
      min_edge -- minimum edge length (None = no resize)
      resize_method -- 'antialias' or 'bicubic'
     
    Outputs:
      imga -- result
    
    """
    
    # -- open image
    img = Image.open(img_fname)#.convert("RGB")

    if max_edge is not None:
        # -- resize so that the biggest edge is max_edge (keep aspect ratio)
        iw, ih = img.size
        if iw > ih:
            new_iw = max_edge
            new_ih = int(round(1.* max_edge * ih/iw))
        else:
            new_iw = int(round(1.* max_edge * iw/ih))
            new_ih = max_edge
        if resize_method.lower() == 'bicubic':
            img = img.resize((new_iw, new_ih), Image.BICUBIC)
        elif resize_method.lower() == 'antialias':
            img = img.resize((new_iw, new_ih), Image.ANTIALIAS)
        else:
            raise ValueError("resize_method '%s' not understood", resize_method)

    # -- convert to a numpy array
    imga = array_from_image(img)
        
    return imga

def array_from_image(img):
    flatten = img.mode not in ['RGB','RGBA','RGBa','CMYK','YCbCr','RGBX']
    imga = sp.misc.fromimage(img,flatten=flatten)
    return imga
    
# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def get_image2(img_fname, resize=None,resize_method=None):
    """ Return a resized image as a numpy array

    Inputs:
      img_fname -- image filename
      resize -- tuple of (type, size) where type='min_edge' or 'max_edge' 
                if None = no resize
     
    Outputs:
      imga -- result
    
    """
    
    # -- open image
    img = Image.open(img_fname)                

    # -- resize image if needed
    if resize is not None:
        rtype, rsize = resize

        if rtype == 'min_edge':
            # -- resize so that the smallest edge is rsize (keep aspect ratio)
            iw, ih = img.size
            if iw < ih:
                new_iw = rsize
                new_ih = int(round(1.* rsize * ih/iw))
            else:
                new_iw = int(round(1.* rsize * iw/ih))
                new_ih = rsize

        elif rtype == 'max_edge':
            # -- resize so that the biggest edge is rszie (keep aspect ratio)
            iw, ih = img.size
            if iw > ih:
                new_iw = rsize
                new_ih = int(round(1.* rsize * ih/iw))
            else:
                new_iw = int(round(1.* rsize * iw/ih))
                new_ih = rsize
        
        else:
            raise ValueError, "resize parameter not understood"

        if resize_method.lower() == 'bicubic':
            img = img.resize((new_iw, new_ih), Image.BICUBIC)
        elif resize_method.lower() == 'antialias':
            img = img.resize((new_iw, new_ih), Image.ANTIALIAS)
        else:
            raise ValueError("resize_method '%s' not understood", resize_method)

    # -- convert to a numpy array
    imga = array_from_image(img)
    
    return imga


def image2array(rep,fobj):
    if rep.get('preproc'):
        resize_type = rep['preproc'].get('resize_type', 'input')
        if resize_type == 'output':
            if 'max_edge' not in rep['preproc']:
                raise NotImplementedError
            # add whatever is needed to get output = max_edge
            new_max_edge = rep['preproc']['max_edge']
    
            preproc_lsum = rep['preproc']['lsum_ksize']
            new_max_edge += preproc_lsum-1
                
            if rep.get('normin'):     
                normin_kshape = rep['normin']['kshape']
                assert normin_kshape[0] == normin_kshape[1]
                new_max_edge += normin_kshape[0]-1
        
            if rep.get('filter'):
                filter_kshape = rep['filter']['kshape']
                assert filter_kshape[0] == filter_kshape[1]
                new_max_edge += filter_kshape[0]-1
            
            if rep.get('normout'):
                normout_kshape = rep['normout']['kshape']
                assert normout_kshape[0] == normout_kshape[1]
                new_max_edge += normout_kshape[0]-1
            
            if rep.get('pool'):
                pool_lsum = rep['pool']['lsum_ksize']
                new_max_edge += pool_lsum-1
    
            rep['preproc']['max_edge'] = new_max_edge
        
        if 'max_edge' in rep['preproc']:
            max_edge = rep['preproc']['max_edge']
            resize_method = rep['preproc']['resize_method']
            imgarr = get_image(fobj, max_edge=max_edge,
                               resize_method=resize_method)
        else:
            resize = rep['preproc']['resize']
            resize_method = rep['preproc']['resize_method']        
            imgarr = get_image2(fobj, resize=resize,
                                resize_method=resize_method)
    else:
        img = Image.open(fobj)
        imgarr = array_from_image(img)
    return imgarr
    
def image_preprocessing(arr,params):

    arr = sp.atleast_3d(arr)

    smallest_edge = min(arr.shape[:2])
    preproc_lsum = params['preproc']['lsum_ksize']
    if preproc_lsum is None:
        preproc_lsum = 1
    smallest_edge -= (preproc_lsum-1)
            
    if params.get('normin'):      
        normin_kshape = params['normin']['kshape']
        smallest_edge -= (normin_kshape[0]-1)

    if params.get('filter'):
        filter_kshape = params['filter']['kshape']
        smallest_edge -= (filter_kshape[0]-1)
        
    if params.get('normout'):
        normout_kshape = params['normout']['kshape']
        smallest_edge -= (normout_kshape[0]-1)
    
    if params.get('pool'):
        pool_lsum = params['pool']['lsum_ksize']
        smallest_edge -= (pool_lsum-1)

    arrh, arrw, _ = arr.shape

    if smallest_edge <= 0 and params['conv_mode'] == 'valid':
        if arrh > arrw:
            new_w = arrw - smallest_edge + 1
            new_h =  int(np.round(1.*new_w  * arrh/arrw))
            print new_w, new_h
            raise
        elif arrh < arrw:
            new_h = arrh - smallest_edge + 1
            new_w =  int(np.round(1.*new_h  * arrw/arrh))
            print new_w, new_h
            raise
        else:
            pass
    
    # TODO: finish image size adjustment
    assert min(arr.shape[:2]) > 0

    # use the first 3 channels only
    orig_imga = arr.astype("float32")[:,:,:3]

    # make sure that we don't have a 3-channel (pseudo) gray image
    if orig_imga.shape[2] == 3 \
            and (orig_imga[:,:,0]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,1]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,2]-orig_imga.mean(2) < 0.1*orig_imga.max()).all(): 
        orig_imga = sp.atleast_3d(orig_imga[:,:,0])

    # rescale to [0,1]
    #print orig_imga.min(), orig_imga.max()
    if orig_imga.min() == orig_imga.max():
        raise MinMaxError("[ERROR] orig_imga.min() == orig_imga.max() "
                          "orig_imga.min() = %f, orig_imga.max() = %f"
                          % (orig_imga.min(), orig_imga.max())
                          )
    
    orig_imga -= orig_imga.min()
    orig_imga /= orig_imga.max()

    # -- color conversion
    # insure 3 dims
    #print orig_imga.shape
    if orig_imga.ndim == 2 or orig_imga.shape[2] == 1:
        orig_imga_new = sp.empty(orig_imga.shape[:2] + (3,), dtype="float32")
        orig_imga.shape = orig_imga_new[:,:,0].shape
        orig_imga_new[:,:,0] = 0.2989*orig_imga
        orig_imga_new[:,:,1] = 0.5870*orig_imga
        orig_imga_new[:,:,2] = 0.1141*orig_imga
        orig_imga = orig_imga_new    


    if params['color_space'] == 'rgb':
        orig_imga_conv = orig_imga
#     elif params['color_space'] == 'rg':
#         orig_imga_conv = colorconv.rg_convert(orig_imga)
    elif params['color_space'] == 'rg2':
        orig_imga_conv = colorconv.rg2_convert(orig_imga)
    elif params['color_space'] == 'gray':
        orig_imga_conv = colorconv.gray_convert(orig_imga)
        orig_imga_conv.shape = orig_imga_conv.shape + (1,)
    elif params['color_space'] == 'opp':
        orig_imga_conv = colorconv.opp_convert(orig_imga)
    elif params['color_space'] == 'oppnorm':
        orig_imga_conv = colorconv.oppnorm_convert(orig_imga)
    elif params['color_space'] == 'chrom':
        orig_imga_conv = colorconv.chrom_convert(orig_imga)
#     elif params['color_space'] == 'opponent':
#         orig_imga_conv = colorconv.opponent_convert(orig_imga)
#     elif params['color_space'] == 'W':
#         orig_imga_conv = colorconv.W_convert(orig_imga)
    elif params['color_space'] == 'hsv':
        orig_imga_conv = colorconv.hsv_convert(orig_imga)
    else:
        raise ValueError, "params['color_space'] not understood"
        
    if  params['preproc'].get('cut'):
        cut_shape = params['preproc']['cut']['ker_shape']
        orig_imga_conv = get_central_slice(orig_imga_conv,cut_shape)
        
    return orig_imga,orig_imga_conv
    
 
def get_central_slice(f,s):
    fshape = f.shape[:2]
    d0 = (fshape[0] - s[0])/2
    d1 = (fshape[1] - s[1])/2
    return f[d0:d0+s[0],d1:d2+s[1]].ravel()


def map_preprocessing(imga0,params): 
    
    if imga0.min() != imga0.max():   #TODO: ASK PEOPLE ABOUT THIS
    
        # -- 0. preprocessing
        #imga0 = imga0 / 255.0
        
        # flip image ?
        if params['preproc'].get('flip_lr') is True:
            imga0 = imga0[:,::-1]
            
        if params['preproc'].get('flip_ud') is True:
            imga0 = imga0[::-1,:]            
        
        # smoothing
        lsum_ksize = params['preproc']['lsum_ksize']
        conv_mode = params['conv_mode']
        if lsum_ksize is not None:
             k = sp.ones((lsum_ksize), 'f') / lsum_ksize             
             imga0 = conv(conv(imga0, k[sp.newaxis,:], conv_mode), 
                          k[:,sp.newaxis], conv_mode)
             
        # whiten full image (assume True)
        if 'whiten' not in params['preproc'] or params['preproc']['whiten']:
            imga0 -= imga0.mean()
            if imga0.std() != 0:
                imga0 /= imga0.std()

    return imga0


def postprocess(Normin,Filtered,Activated,Normout,Pooled,Partially_processed,featsel):     
        
    keys = Pooled.keys()
    keys.sort()
    fvector_l = []
    for cidx in keys:
        if featsel:
            fvector = include_map_level_features(Normin[cidx],
                                             Filtered[cidx],
                                             Activated[cidx],
                                             Normout[cidx],
                                             Pooled[cidx],
                                             Pooled[cidx],
                                             featsel)
        else:
            fvector = Pooled[cidx]
        fvector_l += [fvector]

    if featsel:
        fvector_l = include_image_level_features(Partially_processed,fvector_l,featsel) 
    fvector_l = [sp.concatenate([fvector[:,:,ind] for ind in range(fvector.shape[2])]) for fvector in fvector_l]
    return fvector_l
    
def simple_postprocess(obj):     
        
    keys = obj.keys()
    keys.sort()
    fvector_l = []
    for cidx in keys:
        fvector = obj[cidx]
        fvector_l += [fvector]

    fvector_l = [sp.concatenate([fvector[:,:,ind] for ind in range(fvector.shape[2])]) for fvector in fvector_l]
    return fvector_l    

     
def include_image_level_features(orig_imga,fvector_l,featsel):
    # include grayscale values ?
    f_input_gray = featsel['input_gray']
    if f_input_gray is not None:
        shape = f_input_gray
        #print orig_imga.shape
        fvector_l += [sp.misc.imresize(colorconv.gray_convert(orig_imga), shape).ravel()]

    # include color histograms ?
    f_input_colorhists = featsel['input_colorhists']
    if f_input_colorhists is not None:
        nbins = f_input_colorhists
        colorhists = sp.empty((3,nbins), 'f')
        if orig_imga.ndim == 3:
            for d in xrange(3):
                h = sp.histogram(orig_imga[:,:,d].ravel(),
                                 bins=nbins,
                                 range=[0,255])
                binvals = h[0].astype('f')
                colorhists[d] = binvals
        else:
            raise ValueError, "orig_imga.ndim == 3"
            #h = sp.histogram(orig_imga[:,:].ravel(),
            #                 bins=nbins,
            #                 range=[0,255])
            #binvals = h[0].astype('f')
            #colorhists[:] = binvals

        #feat_l += [colorhists.ravel()]
        fvector_l += [colorhists.ravel()]

    return fvector_l
    

def include_map_level_features(Normin,Filtered,Activated,Normout,Pooled,output,featsel):
    feat_l = []

    # include input norm histograms ? 
    f_normin_hists = featsel['normin_hists']
    if f_normin_hists is not None:
        division, nfeatures = f_norminhists
        feat_l += [rephists(Normin, division, nfeatures)]

    # include filter output histograms ? 
    f_filter_hists = featsel['filter_hists']
    if f_filter_hists is not None:
        division, nfeatures = f_filter_hists
        feat_l += [rephists(Filtered, division, nfeatures)]

    # include activation output histograms ?     
    f_activ_hists = featsel['activ_hists']
    if f_activ_hists is not None:
        division, nfeatures = f_activ_hists
        feat_l += [rephists(Activated, division, nfeatures)]

    # include output norm histograms ?     
    f_normout_hists = featsel['normout_hists']
    if f_normout_hists is not None:
        division, nfeatures = f_normout_hists
        feat_l += [rephists(Normout, division, nfeatures)]

    # include representation output histograms ? 
    f_pool_hists = featsel['pool_hists']
    if f_pool_hists is not None:
        division, nfeatures = f_pool_hists
        feat_l += [rephists(Pooled, division, nfeatures)]

    # include representation output ?
    f_output = featsel['output']
    if f_output and len(feat_l) != 0:
        fvector = sp.concatenate([output.ravel()]+feat_l)
    else:
        fvector = output    
   
    return fvector   


# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def rephists(hin, division, nfeatures):
    """ Compute local feature histograms from a given 3d (width X height X
    n_channels) image.

    These histograms are intended to serve as easy-to-compute additional
    features that can be concatenated onto the V1-like output vector to
    increase performance with little additional complexity. These additional
    features are only used in the V1LIKE+ (i.e. + 'easy tricks') version of
    the model. 

    Inputs:
      hin -- 3d image (width X height X n_channels)
      division -- granularity of the local histograms (e.g. 2 corresponds
                  to computing feature histograms in each quadrant)
      nfeatures -- desired number of resulting features 
     
    Outputs:
      fvector -- feature vector
    
    """

    hin_h, hin_w, hin_d = hin.shape
    nzones = hin_d * division**2
    nbins = nfeatures / nzones
    sx = (hin_w-1.)/division
    sy = (hin_h-1.)/division
    fvector = sp.zeros((nfeatures), 'f')
    hists = []
    for d in xrange(hin_d):
        h = [sp.histogram(hin[j*sy:(j+1)*sy,i*sx:(i+1)*sx,d], bins=nbins)[0].ravel()
             for i in xrange(division)
             for j in xrange(division)
             ]
        hists += [h]

    hists = sp.array(hists, 'f').ravel()    
    fvector[:hists.size] = hists
    return fvector
