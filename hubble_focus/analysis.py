from astropy import time as atime
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import datetime
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os

from collections import OrderedDict

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from . import focusmodel

class Analysis(object):
    ''' Create an object that queries the focus archive,
    holds the outputs, and allows for easy analysis. Essentially,
    a set of dictionaries with plotting capabilities.

    Use as follows:

    >>>mydata = FocusData(2009.01.01,2015.01.02) #times between Jan 1st 2009 and Jan 2nd 2015

    This creates the following dictionaries:

    >>>mydata.fitpsf #raw outputs of fitpsf
    >>>mydata.temps #raw temperatures during this time period
    >>>mydata.moves #all SM moves for all time
    >>>mydata.visit #visit-level means and stdevs, derived from mydata.fitpsf and calculated on the fly

    The fitpsf and visit dicts also include entries for the breathing-only,
    full model, and accumulated SM moves.

    Model products are generated at the time of object creation! These should
    represent the latest version of the model.

    To plot the typical suite of figures for regular reporting:

    mydata.plot_suite(line=True)

    '''
    def __init__(self, thermal_path='/grp/hst/OTA/thermal'):
        ''' 
        Parameters:
            start : str, opt.
                Start date, as YYYY.MM.DD. Defaults to 2009.01.01. When
                querying the temperature table, the provided date - 1 is
                adopted for breathing model and interpolation purposes.
            end : str, opt.
                End date, as YYYY.MM.DD. If not specified, queries up
                to latest data in the database.
        '''
        self.thermal_path = thermal_path

        #Query the archive
        log.info('Querying the archive...')
        self.fitpsf = _add_all_psfs()
        self.moves = focusmodel.get_mirror_moves()

        #Apply the model and get accumulated shrinkage
        log.info('Generating model...')
        self._get_model()
        self._get_sm_moves()

        #get confocality
        self._get_confocality()

        #subtract the model here!!!!!!!
        self.fitpsf.update({'breathing_sub' : self.fitpsf['dSM'] - self.fitpsf['orbital_model']})
        self.fitpsf.update({'fullmodel_sub' : self.fitpsf['dSM'] - self.fitpsf['full_model']})

        #Create visit-level products
        log.info('Creating visit-level products...')
        self.visit = self._visit_means()

    def _get_confocality(self):
        ''' Interpolate (at the camera level) the individual ACS and UVIS
        data points onto a common time sampling and then take the difference.
        The visit-level means of this difference then give you the confocality
        for a given visit.


        NB: This samples onto ACS times. Does this create any issues?
        Any time we have ACS data, we should have UVIS data nearby, so I think
        it should be fine. The opposite wouldn't be true--sometimes UVIS data is
        taken without corresponding ACS, right? So that could create strange issues.
        Regardless, need to be careful with this.
        '''

        acs, uvis = self.split_cam_chip(self.fitpsf,'camera')

        #take a temporal mean first (to get unique (time, focus) mapping)
        acs_mean = self.temporal_mean(self.fitpsf['mjd'][acs],self.fitpsf['dSM'][acs])
        uvis_mean = self.temporal_mean(self.fitpsf['mjd'][uvis],self.fitpsf['dSM'][uvis])
        #interpolate uvis onto acs
        f = interp1d(uvis_mean[:,0],uvis_mean[:,1],kind='linear')
        uvisdSM_acsmjd = f(acs_mean[:,0])

        diff = acs_mean[:,1] - uvisdSM_acsmjd #ACS - UVIS on ACS mjd sampling

        #since diff is a unique mapping into time, it's not consistent with the structure of fitpsf or visit dictionaries
        self._confocality = {
                            'mjd' : acs_mean[:,0],
                            'confocality' : diff,
                            'acs_mean' : acs_mean[:,1],
                            'uvis_mean' : uvisdSM_acsmjd}

        #appropriately average and then add to visit dictionary
        allt = []
        alld  = []
        allu = []
        alla = []
        for t in acs_mean[:,0]: #loop through all times
            day = np.isclose(t,acs_mean[:,0],atol=1.,rtol=0.) #all times within 1 day
            tmean = np.mean(acs_mean[:,0][day]) #avg times within that day
            dmean = np.mean(diff[day]) #avg confocality within that day
            umean = np.mean(uvisdSM_acsmjd[day])
            amean = np.mean(acs_mean[:,1][day])
            allt.append(tmean)
            alld.append(dmean)
            allu.append(umean)
            alla.append(amean)

        #this produces multiple copies for each day, so eliminate
        _,unique = np.unique(allt,return_index=True)

        self._confocality_mean = {'mjd' : np.array(allt)[unique],
                                  'confocality' : np.array(alld)[unique],
                                  'uvis_mean' : np.array(allu)[unique],
                                  'acs_mean' : np.array(alla)[unique],
                                  }

    def _get_model(self):
        ''' Takes the dictionary returned by query.fitpsf()
        and adds entries for the breathing-only and full model,
        as well as accumulated SM moves interpolated onto the
        points given by focusdict['mjd'].

        Returns:
            Nothing. Updates dictionaries in place.

        '''
        times = self.fitpsf['mjd']
        secular = focusmodel.fullmodel(times, model_type = 'secular', add_sm_steps = False, thermal_path = self.thermal_path)
        orbital = focusmodel.fullmodel(times, model_type = 'orbital', add_sm_steps = False, thermal_path = self.thermal_path)
        sm_steps = focusmodel.accumulated(times)

        newentries = {'orbital_model' : orbital,
                      'secular_model' : secular,
                      'full_model' : secular + orbital,
                      'orbital_model_steps' : orbital - sm_steps,
                      'secular_model_steps' : secular - sm_steps,
                      'full_model_steps' : secular + orbital - sm_steps}

        #add to fitpsf dictionary as new keys
        self.fitpsf.update(newentries)

    def _get_sm_moves(self):
        ''' Adds entries to fitpsf dictionary tracking the accumulated
        mirror moves at a given time. Adds key 'accumSM'.
        '''

        secMoveMJD = {}
        for k in self.moves.keys(): secMoveMJD[self._toMJD(k)] = self.moves[k]

        dates = self.fitpsf['mjd']
        accum = np.zeros(len(dates))
        for k,f in secMoveMJD.items():
            ind = np.where(dates < k)
            accum[ind] += f

        self.fitpsf.update({'accumSM':accum})

    def _visit_means(self):
        '''Construct a dictionary of the visit-level means and errors

        Parameters:
            focusdict : dict
                Dictionary returned by query.fitpsf() or get_model

        Returns:
            visitdict : dict
                Dictionary with a single entry per camera per visit.
        '''

        #uniquely identify data from single instrument and visit
        parsed = np.array([n[:6] for n in self.fitpsf['dataset']])
        unique = set(parsed)
        visits = [np.where(parsed == u) for u in unique]

        #grab keys from fitpsf dictionary but drop those for which means, std don't make sense
        columns = list(self.fitpsf.keys())
        drop = ['DOY','camchip','dataset','date','targ','timestamp']
        columns = [c for c in columns if c not in drop]

        #find means, std (after removing Nones)
        visit_means = {c+'_mean' : [] for c in columns}
        visit_means.update({c+'_std' : [] for c in columns})
        visit_means.update({d  : [] for d in drop})
        nonecheck = lambda x : x is not None
        for v in visits: #average numeric data
            for c in columns:
                visit_means[c+'_mean'].append(np.nanmean(list(filter(nonecheck,self.fitpsf[c][v]))))
                visit_means[c+'_std'].append(np.nanstd(list(filter(nonecheck,self.fitpsf[c][v]))))
        for v in visits: #get representative value for non-numeric
            for d in drop:
                visit_means[d].append(self.fitpsf[d][v][0])

        #Now turn all lists into ndarrays
        for c in visit_means.keys():
            visit_means[c] = np.array(visit_means[c])

        return visit_means

    def split_cam_chip(self,dictionary,type):
        ''' Return the indices associated with a
        particular camera/chip.

        Parameters:
            dictionary : dict
                Which dictionary to pull the indices of. Generally
                will be self.fitpsf or self.visit. Must have a 'camchip'
                key.
            type : str
                Choose between splitting at the level of camera
                or the level of chip. 'camera' returns entries for 
                ACS and UVIS (with no chip distinction); 'chip' returns
                entries for ACS1, ACS2, UVIS1, UVIS2.
        '''

        cam = np.array([c[:3] for c in dictionary['camchip']])
        acs = (cam == 'ACS')
        uvis = (cam == 'WFC')

        assert type in ['camera','chip'], 'Invalid type! "camera" or "chip" are the only valid entries.'

        if type == 'camera':
            return acs, uvis
        elif type == 'chip':
            chip = np.array([c[-1] for c in dictionary['camchip']])
            chip1 = (chip == '1')
            chip2 = (chip == '2')

            acs1 = acs & chip1
            acs2 = acs & chip2
            uvis1 = uvis & chip1
            uvis2 = uvis & chip2

            return acs1, acs2, uvis1, uvis2

    def plot_suite(self,line=False,figsize=(10,5),figures=[1,2,3,4]):
        ''' Generate the typical suite of plots for regular reporting. You'll want to run this function
        for each baseline you're interested in (2009-present, 2013-present,2014-present, etc), each of 
        which should be a separate instantiation of this class.

        This is really just a convenient wrapper around the other plotting tools.

        Parameters:
            line : bool, optional
                If True, fit and plot a line to the data. Add text indicating slope and current defocus.
        '''

        cam = np.array([c[:3] for c in self.fitpsf['camchip']])
        vcam = np.array([c[:3] for c in self.visit['camchip']])
        acs = np.where(cam == 'ACS')
        vacs = np.where(vcam == 'ACS')
        uvis = np.where(cam == 'WFC')
        vuvis = np.where(vcam == 'WFC')
        mjd_sort = np.argsort(self.fitpsf['mjd'])

        #plot just the raw points (means and individual)
        if 1 in figures:
            #SM moves? line fit?
            ylim = [self.fitpsf['dSM'].min(),self.fitpsf['dSM'].max()]
            plt.figure(figsize=figsize)
            self.plot_mirror_move(self.fitpsf['mjd'].min(),self.fitpsf['mjd'].max(),
                head_width=26,width=0.4,head_length=0.7)
            line_acs = self.plot_time(self.fitpsf['mjd'][acs],self.fitpsf['dSM'][acs],
                color=[0.7,0.8,0.8],size=30)
            line_uvis = self.plot_time(self.fitpsf['mjd'][uvis],self.fitpsf['dSM'][uvis],
                color=[0.8,0.7,0.8],size=30)
            line_vacs = self.plot_time(self.visit['mjd_mean'][vacs],self.visit['dSM_mean'][vacs],
                color='c',edgecolor='k')
            line_vuvis = self.plot_time(self.visit['mjd_mean'][vuvis],self.visit['dSM_mean'][vuvis],
                color='m',edgecolor='k',
                ylabel='SM Defocus [$\mu$m]',title='Measured Defocus',ylim=ylim)
            plt.legend([line_vacs,line_vuvis],['ACS','UVIS'])

        #plot the breathing-correct raw points (means and individual)
        if 2 in figures:
            plt.figure(figsize=figsize)
            self.plot_mirror_move(self.fitpsf['mjd'].min(),self.fitpsf['mjd'].max(),
                head_width=26,width=0.4,head_length=0.7)
            line_acs = self.plot_time(self.fitpsf['mjd'][acs],self.fitpsf['breathing_sub'][acs],
                color=[0.7,0.8,0.8],size=30)
            line_uvis = self.plot_time(self.fitpsf['mjd'][uvis],self.fitpsf['breathing_sub'][uvis],
                color=[0.8,0.7,0.8],size=30)
            line_vacs = self.plot_time(self.visit['mjd_mean'][vacs],self.visit['breathing_sub_mean'][vacs],
                color='c',edgecolor='k')
            line_vuvis = self.plot_time(self.visit['mjd_mean'][vuvis],self.visit['breathing_sub_mean'][vuvis],
                ylabel='SM Defocus [$\mu$m]',title='Breathing-Corrected Defocus',color='m',edgecolor='k',ylim=ylim)
            plt.legend([line_vacs,line_vuvis],['ACS','UVIS'])

        #raw with SM moves removed
        if 3 in figures:
            ylim = [(self.fitpsf['dSM']+self.fitpsf['accumSM']).min(),(self.fitpsf['dSM']+self.fitpsf['accumSM']).max()]
            plt.figure(figsize=figsize)
            self.plot_mirror_move(self.fitpsf['mjd'].min(),self.fitpsf['mjd'].max(),
                head_width=26,width=0.4,head_length=0.7)
            line_acs = self.plot_time(self.fitpsf['mjd'][acs],self.fitpsf['dSM'][acs] + self.fitpsf['accumSM'][acs],
                color=[0.7,0.8,0.8],size=30)
            line_uvis = self.plot_time(self.fitpsf['mjd'][uvis],self.fitpsf['dSM'][uvis] + self.fitpsf['accumSM'][uvis],
                color=[0.8,0.7,0.8],size=30)
            line_vacs = self.plot_time(self.visit['mjd_mean'][vacs],self.visit['dSM_mean'][vacs] + self.visit['accumSM_mean'][vacs],
                color='c',edgecolor='k')
            line_vuvis = self.plot_time(self.visit['mjd_mean'][vuvis],self.visit['dSM_mean'][vuvis] + self.visit['accumSM_mean'][vuvis],
                ylabel='SM Defocus [$\mu$m]',title='Accumulated Measured Defocus',color='m',edgecolor='k',ylim=ylim)
            if line:
                #linefit performed with both acs and uvis
                line_x, line_y, slope = self.line_fit(self.fitpsf['mjd'],self.fitpsf['dSM'] + self.fitpsf['accumSM'])
                line_leg = 'Linear Fit'
                current = line_y[-1]
                line = self.plot_line(line_x,line_y)
                plt.legend([line_vacs,line_vuvis,line],['ACS','UVIS',line_leg])
                plt.text(0.99,0.75,'Slope: {:+.2f} $\mu m$/year'.format(slope*365),
                    fontsize=12,transform=plt.gca().transAxes, horizontalalignment='right')
                plt.text(0.99,0.68,'Current focus: {:+.2f} $\mu m$'.format(current),
                    fontsize=12,transform=plt.gca().transAxes, horizontalalignment='right')
            else:
                plt.legend([line_vacs,line_vuvis],['ACS','UVIS'])

        #breathing-corr with SM moves removed
        if 4 in figures:
            plt.figure(figsize=figsize)
            self.plot_mirror_move(self.fitpsf['mjd'].min(),self.fitpsf['mjd'].max(),
                head_width=26,width=0.4,head_length=0.7)
            line_acs = self.plot_time(self.fitpsf['mjd'][acs],self.fitpsf['breathing_sub'][acs] + self.fitpsf['accumSM'][acs],
                color=[0.7,0.8,0.8],size=30)
            line_uvis = self.plot_time(self.fitpsf['mjd'][uvis],self.fitpsf['breathing_sub'][uvis] + self.fitpsf['accumSM'][uvis],
                color=[0.8,0.7,0.8],size=30)
            line_vacs = self.plot_time(self.visit['mjd_mean'][vacs],self.visit['breathing_sub_mean'][vacs] + self.visit['accumSM_mean'][vacs],
                color='c',edgecolor='k')
            line_vuvis = self.plot_time(self.visit['mjd_mean'][vuvis],self.visit['breathing_sub_mean'][vuvis] + self.visit['accumSM_mean'][vuvis],
                ylabel='SM Defocus [$\mu$m]',title='Accumulated Breathing-Corrected Defocus',color='m',edgecolor='k',ylim=ylim)
            if line:
                #linefit performed with both acs and uvis
                line_x, line_y, slope = self.line_fit(self.fitpsf['mjd'],self.fitpsf['breathing_sub'] + self.fitpsf['accumSM'])
                line_leg = 'Linear Fit'
                current = line_y[-1]
                line = self.plot_line(line_x,line_y)
                plt.legend([line_vacs,line_vuvis,line],['ACS','UVIS',line_leg])
                plt.text(0.99,0.75,'Slope: {:+.2f} $\mu m$/year'.format(slope*365),
                    fontsize=12,transform=plt.gca().transAxes, horizontalalignment='right')
                plt.text(0.99,0.68,'Current focus: {:+.2f} $\mu m$'.format(current),
                    fontsize=12,transform=plt.gca().transAxes, horizontalalignment='right')
            else:
                plt.legend([line_vacs,line_vuvis],['ACS','UVIS'])

    def plot_confocality(self,figsize=(10,5)):
        ''' Plot confocality '''

        plt.figure(figsize=figsize)
        self.plot_time(self._confocality['mjd'],self._confocality['confocality'],color=[0.7,0.8,0.8],size=30)
        self.plot_time(self._confocality_mean['mjd'],self._confocality_mean['confocality'],color='c',edgecolor='k',
            ylabel='ACS - UVIS dSM [$\mu$m]',title='Confocality',)

        self.plot_mirror_move(self.fitpsf['mjd'].min(),self.fitpsf['mjd'].max(),
            head_width=26,width=0.4,head_length=0.7)

    def plot_line(self,x,y,c='k',ls='--',lw=3):
        ''' Plot a dashed line given by x,y '''

        ind = np.argsort(x)
        line, = plt.plot(x[ind],y[ind],linestyle=ls,lw=lw,c=c)

        return line

    def line_fit(self,x,y):
        ''' Fit a line to some x,y '''

        line = lambda x, a, b: a*x + b
        poly, cov = curve_fit(line,x,y)

        x = np.arange(x.min()-50,x.max()+50)
        line = x*poly[0] + poly[1]

        return x, line, poly[0]

    def spatial_mean(self,x,y,z):
        ''' Pass in the x,y coordinates and take the mean z for at that point.
        The output takes on the form [x,y,mean_z,std_z]. In order to plot, for example,
        one might call plt.plot(spatial[:,0],spatial[:,1],c=spatial[:,2])

        Parameters:
            x,y,z : nd arrays

        Returns:
            spatial : nd array
                Array for which the columns are x,y,mean_z,std_z.
        '''

        xy = zip(x,y)
        nonecheck = lambda x : x is not None

        spatial = []
        for i in set(xy):
            field_ind = (np.array(xy) == i)[:,0]
            mean = np.nanmean(list(filter(nonecheck,z[field_ind])))
            std = np.nanstd(list(filter(nonecheck,z[field_ind])))
            tmp = list(i)
            tmp.extend([mean, std])
            spatial.append(tmp)

        return np.array(spatial)

    def temporal_mean(self,time,y):
        ''' Take the mean of some value y at a given time t. Note that this
        differs from the visit-level mean. The output follows that of spatial_mean.

        You might consider expanding this to reflect _visit_means() and construct
        a dictionary with the temporal mean computed for all numeric values, but
        that complicates the matter if the user desires to average chips1 and 2
        separately, for example. This function provides more flexibility.

        Parameters:
            t, y: nd arrays

        Returns:
            temp : nd array
                Array for which the columns are t,mean_y,std_y
        '''

        nonecheck = lambda x : x is not None

        temp = []
        for t in set(time):
            temp_ind = (time == t)
            mean = np.nanmean(list(filter(nonecheck,y[temp_ind])))
            std = np.nanstd(list(filter(nonecheck,y[temp_ind])))
            tmp = [t]
            tmp.extend([mean, std])
            temp.append(tmp)
        return np.array(temp)

    def plot_accum_shrinkage(self,include_archival=True,show_moves=True):
        ''' Plot the long-term, accumulated shrinkage of the OTA. It 
        really only makes sense to plot this if you've queried the
        entire focus database.

        Parameters:
            include_archival: bool, opt.
                Include pre-2009 data? Default True.
            show_moves : bool, opt.
                Plot SM moves with arrows? Default True.
        '''

        focus = self.visit['dSM_mean'] #non-corrected, since archival data doesn't have good correction
        mjd = self.visit['mjd_mean']

        if include_archival:
            old = np.loadtxt('archivaldata.txt',skiprows=2,usecols=[2,3])
            focus = np.concatenate((old[:,1],focus))
            mjd = np.concatenate((old[:,0],mjd))

        #re-do accumulated shrinkage calculation, because the ancient (pre-2009) data needs this correction too.
        secMoveMJD = {}
        for k in self.moves.keys(): secMoveMJD[self._toMJD(k)] = self.moves[k]
        accum = np.zeros_like(mjd)
        for k,f in secMoveMJD.items():
            ind = np.where(mjd < k)
            accum[ind] += f
        focus += accum

        x_range = (mjd.min(),mjd.max())

        plt.figure(figsize=(16,8))
        plt.scatter(mjd,focus,c='c',s=40,edgecolor='None')
        plt.hlines(0,x_range[0],x_range[1],linestyles='dashed')

        times = np.arange(48000,x_range[1],365*3)
        dates = [atime.Time(t,format='mjd',scale='utc') for t in times]
        plt.xticks(times,[d.datetime.strftime('%b %Y') for d in dates],fontsize=14,rotation=45)
        plt.yticks(fontsize=14)

        plt.xlim([48000,x_range[1]])
        plt.ylim([-20,160])

        plt.ylabel('Accumulated OTA Shrinkage [$\mu$m]',fontsize=16)

        if show_moves:
            self.plot_mirror_move(times[0],times[-1])

        plt.show()

    def plot_mirror_move(self,start,end,head_width=40,width=0.2,head_length=2.):
        ''' Generate arrows representing SM mirror moves

        Parameters: 
            start : float
                MJD value for the lower limit of SM moves to plot
            end : float
                MJD value for the upper limit of SM moves to plot
            head_width : float, opt.
                Width of arrow head
            width : float, opt.
                Width of arrow tail
            head_length : float, opt.
                Length of arrow head

        Returns:
            Nothing
        '''
        secMoveMJD = {}
        for k in self.moves.keys(): secMoveMJD[self._toMJD(k)] = self.moves[k]
        for time,move in secMoveMJD.items():
            if time > start and time < end:
                plt.arrow(time,0,0,move,width=width,head_width=head_width,head_length=head_length,fc='k',
                    length_includes_head=True,overhang=0.5)

    def scatter_plot(self,x,y,color=None,ylabel='',xlabel='',title='',alpha=1,size=50,edgecolor='None',zorder=1,ylim=None):
        ''' Standard function for plotting. Essentially just enforces
        a particular matplotlib style upon plots produced.

        Parameters:
            x : nd array
                x data
            y : nd array
                y data
            color : str or 3-tuple
                matplotlib color for markers -- 'c', 'k', etc. or RGB value [0.7,0.5,0.3]
            ylabel : str, opt.
                Label for y-axis
            xlabel : str, opt.
                Label for x-axis
            title : str, opt.
                Title of plot
            alpha : float, opt.
                Value between 0 and 1 for marker transparency
            size : float or int, opt.
                Marker size
            edgecolor : str or 3-tuple, opt.
                Marker edges, see color parameter
            zorder : int, opt.
                Ordering of markers, useful when plotting multiple times to a single figure
            ylim : tuple, opt.
                Y-axis limits

        Returns:
            matplotlib Line2D object
        '''

        line = plt.scatter(x,y,c=color,s=size,edgecolor=edgecolor,alpha=alpha,zorder=zorder)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel(xlabel,fontsize=16)
        plt.ylabel(ylabel,fontsize=16)
        plt.title(title,fontsize=18)

        return line


    def plot_time(self,x,y,color=None,ylabel='',xlabel='',title='',alpha=1,size=50,edgecolor='None',zorder=1,ylim=None):
        ''' Standard function for plotting against time. Assumes you're passing
        in time as an MJD value, which is then converted into a human readable month, year
        when plotted.

        Parameters:
            x : nd array
                Time, expected as mjd.
            y : nd array
                Some function of time.
            color : str or 3-tuple
                matplotlib color for markers -- 'c', 'k', etc. or RGB value [0.7,0.5,0.3]
            ylabel : str, opt.
                Label for y-axis
            xlabel : str, opt.
                Label for x-axis
            title : str, opt.
                Title of plot
            alpha : float, opt.
                Value between 0 and 1 for marker transparency
            size : float or int, opt.
                Marker size
            edgecolor : str or 3-tuple, opt.
                Marker edges, see color parameter
            zorder : int, opt.
                Ordering of markers, useful when plotting multiple times to a single figure
            ylim : tuple, opt.
                Y-axis limits

        Returns:
            Nothing
        '''
        start = x.min()
        end = x.max()

        line = self.scatter_plot(x,y,color=color,ylabel=ylabel,xlabel=xlabel,title=title,alpha=alpha,
            size=size,edgecolor=edgecolor,zorder=zorder,ylim=ylim)
        plt.hlines(0,start-50,end+50,linestyles='dashed')

        times = np.arange(start,end,365)
        dates = [atime.Time(t,format='mjd',scale='utc') for t in times]
        plt.xticks(times,[d.datetime.strftime('%b %Y') for d in dates],fontsize=14,rotation=45)

        plt.xlim([start-50,end+50])
        plt.ylim(ylim)
        return line

    def _toMJD(self,x):
        ''' Convert str date 'YYYY.MM.DD' to MJD '''
        return atime.Time(datetime.datetime.strptime(x,'%Y.%m.%d'),scale='utc').mjd



def add_visit(dirname, alldict):
    '''Get resultsChip1.txt and resultsChip2.txt
    from latest visit and append to database.

    If you want to update values from a visit already
    ingested, delete the old ones and simply use this
    function to append the updated values.

    Parameters:
        dirname : str
            Directory containing data to append/update. Looks for
            resultsChip1.txt and resultsChip2.txt
    '''

    keys = list(alldict.keys())

    directory = os.path.join(dirname,'resultsChip[1,2].txt')
    files = glob(directory)

    for f in files:
        log.info('Adding: {}'.format(f))
        parsed = _parse_fitpsf(f)
        for line in parsed:
            for i, key in enumerate(keys):
                try:
                    alldict[key].append(line[i])
                except IndexError: #not all files/lines are completely populated
                    alldict[key].append(None)

def _add_all_psfs():
    '''Go back as far as individual fitpsf outputs exist
    and read them in. In some old data have outputs, but
    without all columns. Might be able to use, just with
    Null entries.

    This should really only be run when initializing the database.
    '''

    prop_paths = np.asarray(sorted(glob('/grp/hst/OTA/focus/Data/prop*')))
    props = np.asarray([int(p.split('prop')[1]) for p in prop_paths])
    
    vislist = []
    for p in prop_paths[props >= 11877]: #SM4 to present:
        tmp = glob(os.path.join(p,'visit*/'))
        vislist.extend(tmp)

    columns = ['camchip','x','y','background','dataset','targ','mjd','date','timestamp','DOY',
        'dSM','z4','xcoma','ycoma','xastig','yastig','spher','xclov','yclov','xsphash',
        'ysphash','xash','yash','spher5th','fitbg','xtilt','ytilt','blur']
    allvisits = OrderedDict([(key, []) for key in columns])

    for v in vislist:
        add_visit(v, allvisits)

    for k, v in allvisits.items():
        allvisits[k] = np.asarray(v)

    return allvisits

def _parse_fitpsf(fname):
    ''' Parse an individual resultsChip#.txt file from
    fitpsf return as a list of tuples
    '''
    fh = open(fname)
    out = []
    blurstr = {'blur':23}
    cameralist = ['ACSWFC1','ACSWFC2','WFC3UVIS1','WFC3UVIS2']
    cameralist.append('ACSWFS2') # prop12780, visit11-sept2012 has wrong camera listed in chip2


    dtypes = [str,int,int,float,str,str,float,str,str] + [float]*19

    #Parse file
    for line in fh.readlines():
        for camera in cameralist:
            isheader = camera in line
            if isheader:
                tmp = line.strip().split()
                x = tmp[1]
                y = tmp[2]
                background = tmp[3]
                cam = camera
                break

        if not isheader:

            tmp = line.strip().split()

            #Two values are joined in ACS output when the second is negative. Split them
            if len(tmp) == 23:
                both = tmp[-7].split('-')
                both[1] = '-'+both[1]
                
                tmp.pop(-7)
                tmp.insert(-6,both[0])
                tmp.insert(-6,both[1])

            #Treat inf properly
            for i,t in enumerate(tmp):
                if t == '*********': tmp[i] = -1

            #Add info from header line
            tmp[:0] = [cam,x,y,background]

            #Convert to proper datatypes
            num = lambda x, y: x(y)
            tmp = [num(d,e) for d,e in zip(dtypes,tmp)]

            out.append(tmp)

    #Eliminate duplicates
    out_tuple = [tuple(l) for l in out]
    out_set = set(out_tuple)

    eliminated = len(out_tuple) < len(out)
    if eliminated != 0:
        log.info('Eliminated {} duplicate outputs'.format(eliminated))

    #Sort by date and x-pos
    outlist = list(out_set)
    outlist.sort(key=lambda x: (x[5],x[1]))

    fh.close()
    return outlist