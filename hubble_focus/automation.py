# A collection of tools for automating fitpsf
from copy import deepcopy
import multiprocessing as mp
import os, subprocess, shutil

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import numpy as np
from photutils import aperture_photometry, daofind, CircularAnnulus, DAOStarFinder
from scipy import spatial
from scipy.optimize import leastsq, least_squares

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Automated Source Finding ---

class FocusImage(object):
    '''
    '''
    def __init__(self,filepath,extension=1):
        self.fits_file = fits.open(filepath)
        self.image = self.fits_file[extension].data
        self.image_header = self.fits_file[extension].header
        self.fits_header = self.fits_file[0].header
        self.extension = extension
        self.filepath = filepath
        
        self.sources = None
        self.backgrounds = None
        self.mean = None
        self.median = None
        self.std = None

    def find_sources(self,clobber=False,*args,**kwargs):
        '''See find_sources for documentation'''
        sources, mean, median, std = find_sources(self.image,
                                                  mean = self.mean,
                                                  median = self.median,
                                                  std = self.std,
                                                  *args,**kwargs)
        self.mean = mean
        self.median = median
        self.std = std
        if self.sources is None or clobber:
            self.sources = sources
        else: #add unique sources in
            alreadyhave,_ = find_common_sources(sources,self.sources,
                                                return_indices=True,
                                                maxsep=5)
            self.sources = np.vstack((self.sources,sources[~alreadyhave]))
            
    def find_backgrounds(self,*args,**kwargs):
        '''See find_backgrounds for documentation'''
        assert self.sources is not None, 'No sources to find backgrounds for!'
        self.backgrounds = find_backgrounds(self.image,self.sources,
                                            *args,**kwargs)
        
    def extract_psf(self,source_index,*args,**kwargs):
        return extract_psf(self.image,self.sources[source_index],
                           *args,**kwargs)

    def display_image(self, figsize=(10, 5)):
        shifted = self.image - self.image.min()
        median = np.nanmedian(shifted)
        std = np.nanstd(shifted)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(shifted, cmap='Greys_r' ,vmin=median-0.1*std, vmax=median+0.5*std, interpolation='nearest')
        if self.sources is not None:
            ax.plot(self.sources[:,0], self.sources[:,1], 'o', mfc='none', ms=4) #facecolors='none'
            ax.set_xlim(0,self.image.shape[1])
            ax.set_ylim(0,self.image.shape[0])
        return fig, ax        

def find_sources(image,sigma=5.,iters=1,threshold=50.,fwhm=2.,isolation=30.,edge_distance=15.,sigma_radius=1.5,
    mean=None,median=None,std=None,xyslice=None,roundlo=-1.,roundhi=1.,peak_threshold=None):
    '''
    Find sources in an iamge with DAOStarFinder and eliminate any that don't meet
    some isolation criterion and any that fall too close the edge of the image.
    
    Parameters:
        image : nd array
            2D star field
        sigma : float
            Sigma for performing sigma-clipped stats (primary for std dev and median
            determination)
        iters : int
            Number of iterations of sigma-clipping
        fwhm : float
            FWHM (in pixels) of stellar sources
        threshold : float
            Number of standard deviations from sky median sources must be
            to qualify as good sources
        isolation : float
            Distance in pixels. Any sources that aren't separated by at least this
            distance are eliminated
        edge_distance : float
            Distance in pixels from the edge of the image. Any sources within this
            distance are eliminated
    
    Returns:
        good_sources : nd array
             Array of (x,y) tuple source coordinates
    '''
    if xyslice is None:
        xyslice = [slice(None,None),slice(None,None)]
    #full image stats followed by first pass at star finding
    if mean is None or median is None or std is None:
        mean, median, std = sigma_clipped_stats(image[xyslice],sigma=sigma,iters=iters)
    finder = DAOStarFinder(threshold=threshold*std,fwhm=fwhm,sigma_radius=sigma_radius,
                           roundlo=roundlo,roundhi=roundhi)
    sources = finder(image-median)
    
    #eliminate sources that don't meet some isolation criterion
    xy = np.array(list(zip(sources['xcentroid'].data,sources['ycentroid'].data)))
    pairwise_dist = spatial.distance.cdist(xy,xy)
    diag_indices = np.diag_indices(pairwise_dist.shape[0])
    pairwise_dist[diag_indices] = isolation + 1 #don't exclude based on comparison to self
    isolated_inds = np.all(pairwise_dist > isolation, axis=0)
    xy = xy[isolated_inds]

    #eliminate sources within 15 pixels of the edge
    lowerleft = np.any((xy <= edge_distance),axis=1)
    upper = xy[:,1] >= image.shape[0] - edge_distance
    right = xy[:,0] >= image.shape[1] - edge_distance
    good_inds = lowerleft | upper | right
    good_sources = xy[~good_inds]

    if peak_threshold is not None:
        to_keep = np.asarray([np.nanmax(extract_psf(image, source) > peak_threshold) for source in good_sources])
        good_sources = good_sources[to_keep]

    return good_sources, mean, median, std
    
def find_backgrounds(image,sources,inner_radius=10.,outer_radius=15.):
    '''
    Return the backgrounds of multiple sources
    determine by centering an annulus on each source
    and taking a mean.
    
    Parameters:
        image : nd array
            2D stellar image
        sources : nd array
            Array of (x,y) coordinates
        inner_radius : float
            Radial distance (pixels) of inner annulus
        outer_radius : 
            Radial distance (pixels) of outer annulus
            
    Returns:
        backgrounds : nd array
            Measured background for each star in sources.
    '''
    annulus_apertures = CircularAnnulus(sources, r_in=inner_radius, r_out=outer_radius)
    annulus_phot = aperture_photometry(image, annulus_apertures)
    
    bkg_mean = annulus_phot['aperture_sum'] / annulus_apertures.area()
    backgrounds = bkg_mean.data
    
    return backgrounds

def extract_psf(image,xy,box_size=(15,15)):
    xyslice = (slice(int(np.rint(xy[1]-box_size[1]/2)),int(np.rint(xy[1]+box_size[1]/2))),
               slice(int(np.rint(xy[0]-box_size[0]/2)),int(np.rint(xy[0]+box_size[0]/2))))
    return image[xyslice]

def find_common_sources(sources1,sources2,maxsep=2.,return_indices=False):
    pairwise_dist = spatial.distance.cdist(sources1,sources2)
    # find matches by minimizing distance and then rejecting any minima that are too large
    minind = np.argmin(pairwise_dist,axis=1)
    mindists = pairwise_dist[range(len(sources1)),minind] 
    goodmins = mindists < maxsep
    matches = minind[goodmins] #mapping onto im2, if interested (xy2[matches])
    
    if return_indices:
        return goodmins, matches #indices for 1, 2
    else:
        common1 = sources1[goodmins] # sources in sources1 that have matches in sources2
        common2 = sources2[matches] # sources in sources2 that have matches in sources1
        return common1, common2

def match_many_source_lists(reference,list_of_lists,offset=None,return_indices=False,maxsep=2.0):
    '''
    Return only the sources found in a reference list that
    can also be found in all lists.

    There's probably a cleaner way to do this, but
    the approach here is to make two passes through:
    1. Find the subset of references sources that
    are present in all other lists
    2. Match this subset to the right indices
    in each list in list_of_lists
    '''

    subset = []
    for i,l in enumerate(list_of_lists):
        if offset is not None:
            l = deepcopy(l)
            l += offset[i]
        idx1,_ = find_common_sources(reference,l,return_indices=True,maxsep=maxsep)
        subset.append(idx1)
    common = np.all(subset,axis=0)
    reference_subset = reference[common]
    
    list_indices = []
    for i,l in enumerate(list_of_lists):
        if offset is not None:
            l = deepcopy(l)
            l += offset[i]
        _,list_index = find_common_sources(reference_subset,l,return_indices=True,maxsep=maxsep)
        list_indices.append(list_index)
    
    if return_indices:
        return common, list_indices
    else:
        return reference_subset, [l[idx] for l,idx in zip(list_of_lists,list_indices)]


# --- Automate fitpsf with multiprocessing ---

def multifitpsf(imlist, sourcexy, sourcebg, infile, spatial_funcs = {}, nprocesses=8):
    #Can handle single and multi-image fits.

    # Single image gets placed in single-element list
    if isinstance(imlist, str):
        imlist = [imlist, ]
    imlist = [os.path.splitext(im)[0] for im in imlist] # remove extensions

    # Need to reshape sources and backgrounds to follow: (nstars x nfiles x coords/bg)
    # The expectation is that the user will pass in a list of that follows: (nfiles x nstars x coords/bg)
    # or (nstars x coords/bg)
    sourcexy = np.asarray(sourcexy)
    sourcebg = np.asarray(sourcebg)
    if np.ndim(sourcexy) < 3:
        sourcexy = np.expand_dims(sourcexy,0)
    if np.ndim(sourcebg) < 2:
        sourcebg = np.expand_dims(sourcebg,0)
    sourcexy = sourcexy.swapaxes(0,1)
    sourcebg = sourcebg.swapaxes(0,1)

    #for each process, create a tmp directory
    #in each directory, create a dowfc.pro that
    #changes IDL's working directory and then runs fitpsf
    tmpdirs = []
    scripts = []
    for n in range(nprocesses):
        tmpname = 'process{}'.format(n)
        if os.path.exists(tmpname):
            shutil.rmtree(tmpname)
        os.mkdir(tmpname)
        log.info('Creating temporary directory {}'.format(tmpname))
        
        #copy fits files over
        for im in imlist:
            shutil.copy(im+'.fits',tmpname)
        tmpdirs.append(tmpname)
        
        #make list file
        make_list_file(imlist,os.path.join(tmpname,'list'))
        
        #make batch idl script
        make_idl_script(tmpname,'list',os.path.join(tmpname,'script.pro'))
        
        #save infile
        #with open(os.path.join(tmpname,'list.in'),'w+') as f:
        #    f.write(infile)
            
        #compile list of scripts for each process
        scripts.append(os.path.join(tmpname,'script.pro'))

    #split up the stars evenly among the processes
    #and have them run in each of those directories
    image_chunk = [imlist,] * nprocesses
    par_chunk = [infile,] * nprocesses
    func_chunk = [spatial_funcs,] * nprocesses
    xy_chunk = np.array_split(sourcexy,nprocesses,axis=0)
    bg_chunk = np.array_split(sourcebg,nprocesses,axis=0)
    pool = mp.Pool(processes=nprocesses)
    results = pool.map(worker,zip(scripts,tmpdirs,image_chunk,xy_chunk,bg_chunk,par_chunk,func_chunk))
    pool.close()
    pool.join()
    
    #clean up directories
    for d in tmpdirs:
        shutil.rmtree(d)
        
    stacked_results = np.vstack([r for r in results if len(r) > 0]) # reject any empty results

    # Turn into numpy record array
    newkeys = ['filename','x','y']
    newkeys.extend(list(infile.__dict__.keys())[7:])
    keytype = [stacked_results.T[0].astype(str).dtype,] # get string of right length for filenames
    keytype.extend( [float,] * (len(newkeys) - 1) )
    return np.core.records.fromarrays(stacked_results.T, dtype = [(key, ktype) for key, ktype in zip(newkeys, keytype)])
        
def worker(args):
    return fitpsf_process(*args)
        
def fitpsf_process(script,dirname,images,stars,bgs,parameters,spatial_funcs):
    #Set up .dat for fitpsf. I'm assuming Extracted PSF size will always be 15.
    header = '      15  =  Extracted PSF size (must be odd valued\n       x       y     background\n'

    files = images

    nstars = stars.shape[0]

    log.info('FITS files: ' + str(files))
    log.info('Fitting PSFs for {n} stars.'.format(n = nstars))

    data = []
    #Loop through the stars
    for i,s in enumerate(stars):
        #Produce the .cos (bad pixels) and .dat (xpos,ypos, background) files that mkfocusin expects
        log.info('Producing .cos and .dat files...')
        for j,f in enumerate(files):
            x = s[j][0]
            y = s[j][1]
            background = bgs[i][j]
            
            out = header + '    ' + str(int(np.rint(x))) + '    ' + str(int(np.rint(y))) + '      ' \
                + str(background) + '       0       0\n'
            fbase = os.path.basename(f)
            fcos = open(os.path.join(dirname,fbase)+'.cos','w+')
            fcos.write('') #Pretend no bad pixels for now
            fcos.close()

            fdat = open(os.path.join(dirname,fbase)+'.dat','w+')
            fdat.write(out)
            fdat.close()
            
            if isinstance(parameters,str):
                with open(os.path.join(dirname,'list.in'),'w+') as f:
                    f.write(parameters)
            else:
                parameters = deepcopy(parameters)
                for key, (func,args) in spatial_funcs.items():
                    parameters.__getattr__(key).value = func(x,y,*args)
                #write out
                parameters.to_file(os.path.join(dirname,'list.in'))

        log.info('Fitting star [{x},{y}] ({i}/{nstars})'.format(x=s[j][0],y=s[j][1],i=i+1,nstars=nstars))
        log.info('Running fitpsf...')
        run_idl_script('@'+script)
        #call focusresults
        #log.info('Running focusresults...')
        #run_idl_script('focusresults, {}'.format('dowfc_list'))
        try:
            #Save results
            data.append(parse_par(os.path.join(dirname,'list.par'),len(images)))
            #Remove old .par files to avoid doubling entries in results.txt
            os.remove(os.path.join(dirname,'list.par'))
        except FileNotFoundError:
            log.warning('Couldn\'t find .par file! Fit must have failed. Skipping.')
            noutputs = len(parameters.__dict__) - 7 + 3 # - header + name,x,y
            data.append([np.nan,] * noutputs)
        
    return np.array(data)

def make_list_file(imlist,outpath):
    with open(outpath,'w+') as f:
        text = '\n'.join([os.path.basename(im) for im in imlist])
        f.write(text)
        
def make_idl_script(dirname,listname,outname):
    text = 'cd, "{}"\nfitpsf, "@{}",0,1,/EE'.format(dirname,listname)
    with open(outname,'w+') as f:
        f.write(text)

def run_idl_script(script_file_name):
    subprocess.call(['/Applications/exelis/idl84/bin/idl','-quiet','-e',script_file_name])
    
def parse_par(filename,nstars):
    #can parse both single and multi-fits. Just need mapping
    #of coeffs to position (which is a function of .in files
    #and number of stars being fit simultaneously)
    with open(filename) as f:
        lines = f.read().splitlines()
        xylines = 8
        coefflines = (range(16+nstars,len(lines)))

        filename = lines[xylines].split()[1]
        x = float(lines[xylines].split()[2])
        y = float(lines[xylines].split()[3])
        data = [filename, x, y]
        for c in coefflines:
            data.append(float(lines[c].split()[-3]))
    return np.asarray(data,dtype=object)

def polyfit2d(x,y,z,order=3):
    
    init_coeffs = [0,] * order * order
    result =  least_squares(polylsq, init_coeffs,
                            args = (x,y,z,order),
                            loss = 'soft_l1',
                            f_scale = 0.1)
    return result['x']
    
def polylsq(coeffs,x,y,z,order):
    coeffs = np.asarray(coeffs).reshape(order,order)
    return z -  np.polynomial.polynomial.polyval2d(x,y,coeffs)

def polyval2d(x,y,coeffs):
    order = int(np.sqrt(len(coeffs)))
    square_coeffs = np.asarray(coeffs).reshape(order,order)
    return np.polynomial.polynomial.polyval2d(x,y,square_coeffs)

def polygrid2d(x,y,coeffs):
    order = int(np.sqrt(len(coeffs)))
    square_coeffs = np.asarray(coeffs).reshape(order,order)
    return np.polynomial.polynomial.polygrid2d(x,y,square_coeffs)