# This is a translation of Colin's ModelPieces.py code
# designed to make teasing out the individual model
# components a little easier.

from glob import glob
import os

import datetime
from scipy.interpolate import interp1d
import numpy as np

# -- Model components --

def secular(times):
    '''
    Given a array-like of times in MJD, pass back the secular model.
    '''

    launch = to_julian(1990, 4, 24) # Launch date April 24th 1990

    # Single Exponent mean Fit  A + B exp(-C*days-since-launch) Sami-Matias Niemi 7/21/10
    A = -2.32 - 0.86 + 3.60 + 2.97 + 1.78   # Adjustments from broken line fit of June 2015 and mirror moves
    B = 269.93
    C = 6.366e-4

    extraSlope = -3.31505303e-03
    jbreak = 55643     # 2011-04-13

    cameraMean = 0.324  # Average UVIS and WFC position

    days_since_launch = times - launch
    secular = A + B * np.exp(-C * days_since_launch) + cameraMean 
    # Apply extra slope after 2011-04-13 MJD 55643
    secular[times > jbreak] += extraSlope * (times[times > jbreak] - jbreak)

    return secular

def orbital(times,interpolate_onto_times=False,thermal_path='/grp/hst/OTA/thermal'):
    '''
    Given an array-like of times in MJD, pass back the orbital
    breathing curve.

    Parameters:
        times : tuple or array-like
            Accepts either tuple of (min_time, max_time) or array-like of
            times to interpolate onto.
        interpolate_onto_times : bool, optional
            Interpolate onto the times given? If False,
            returns MJD times in native telemetry sampling.
        thermal_path : str, optional
            By default, we look for telemetry data on central store. This can
            be slow, so circumventing this is recommended.

    Returns:
        mjd : array of times in MJD (only returned if interpolate_onto_times == False)
        breathing : array of defocus values (dSM in microns) at given times 
    '''

    if interpolate_onto_times:
        assert len(times) > 2, 'Cannot interpolate onto {} points'.format(len(times))

    # Get temperatures in native telemetry time sampling
    extra_time = 1. # add a day to both ends of times requested
    start_time = np.min(times) - extra_time
    end_time = np.max(times) + extra_time
    temperatures = get_temperatures(start_time, end_time, path=thermal_path)

    # Get orbital breathing in telemetry time sampling
    mjd, breathing = _thermal_to_breathing(temperatures)

    if interpolate_onto_times:
        # Interpolate onto requested times
        interp_func = interp1d(mjd,breathing,kind='linear')
        return interp_func(times)
    else:
        return mjd, breathing

def _thermal_to_breathing(temperatures):
    '''
    Given temperature recarray from get_temperatures,
    produce the breathing model
    '''

    meanShort = 9.88

    jdate = temperatures['mjd']
    time_sort = np.argsort(jdate) #enforce times correctly ordered
    jdate = jdate[time_sort]
    aftLS = temperatures['aftLS'][time_sort]
    trussAxial = temperatures['trussAxial'][time_sort]
    aftShroud = temperatures['aftShroud'][time_sort]
    fwdShell = temperatures['fwdShell'][time_sort]
    lightShield = temperatures['lightShield'][time_sort]

    # Take the mean of the last 8 lightShield thermal values
    kernel = np.zeros(16)
    kernel[-8:] = 1./8.
    lightShield_mean = np.convolve(lightShield,kernel,mode='same')[8:]
    lag_term = 0.7*(lightShield[8:] - lightShield_mean)
    breathing = 0.48*aftLS[8:] + 0.81*trussAxial[8:] - 0.28*aftShroud[8:] + 0.18*fwdShell[8:] + 0.55*lag_term
    breathing -= meanShort

    return jdate[8:], breathing

def accumulated(times):
    mirror_moves = get_mirror_moves()

    # Loop over mirror moves and add move amount
    # to all times before that date
    accum = np.zeros( len(times) )
    for date,move in mirror_moves.items():
        mjd = to_julian(*[int(d) for d in date.split('.')])
        ind = np.where(times < mjd)
        accum[ind] += move
    return accum

# -- Combined models --

def fullmodel(times,model_type='full',add_sm_steps=False,thermal_path='/grp/hst/OTA/thermal'):
    '''
    Given an array-like of times in MJD, pass back the orbital
    breathing curve.

    Parameters:
        times : array-like
            Times (in MJD) to interpolate model onto. If you're interested
            in seeing the model in the native telemetry time sampling,
            consider calling orbital directly (with a tuple of min and max times)
        model_type : str
            Accepts 'full', 'secular', or 'orbital'. 'full = 'secular' + 'orbital'
        add_sm_steps: bool, opt
            Removes the commanded SM mirror moves, if requested.
        thermal_path : str, optional
            By default, we look for telemetry data on central store. This can
            be slow, so circumventing this is recommended.
    Returns:
        full_model : nd array
            Array of requested model values sampled onto input times.
    '''
    model_types = ['full','secular','orbital']
    assert model_type in model_types, 'model_type not understood! Must be one of {}'.format(model_types)

    if model_type == 'full':
        secular_model = secular(times)
        orbital_model = orbital(times, interpolate_onto_times=True, thermal_path=thermal_path)
        full_model = secular_model + orbital_model
    elif model_type == 'secular':
        full_model = secular(times)
    elif model_type == 'orbital':
        full_model = orbital(times, interpolate_onto_times=True, thermal_path=thermal_path)

    if add_sm_steps:
        full_model -= accumulated(times)

    return full_model

# -- Utility Functions --

def get_mirror_moves():
    return  {u'1994.03.07': 95.0,
             u'1994.06.29': 5.0,
             u'1995.01.15': 5.0,
             u'1995.08.28': 6.5,
             u'1996.03.14': 6.0,
             u'1996.10.30': 5.0,
             u'1997.03.18': -2.4,
             u'1998.01.12': 21.0,
             u'1998.02.01': -18.6,
             u'1998.06.04': 16.6,
             u'1998.06.28': -15.2,
             u'1999.09.15': 3.0,
             u'2000.01.09': 4.2,
             u'2000.06.15': 3.6,
             u'2002.12.02': 3.6,
             u'2004.12.22': 4.16,
             u'2006.07.31': 5.34,
             u'2009.07.20': 2.97,
             u'2013.01.24': 3.6,
             u'2013.11.12': 2.97,
             u'2015.02.05': 1.78}

def _parse_temperature(fname):
    ''' Parse a .fof file and return
    list of mjd with temperatures
    '''
    with open(fname) as data:
        out = []
        for line in data.readlines():
            tmp = _parse_temp_line(line)
            out.append(tmp)
    return out

def _parse_temp_line(line):
    ''' Parse a single line from a 
    .fof file
    '''
    dtypes = [float]*7
    tmp = line.strip().split()[1:] # toss out timestamp, since we already have MJD
    #Convert to proper datatypes
    num = lambda x, y: x(y)
    tmp = [num(d,e) for d,e in zip(dtypes,tmp)]

    return tmp

def get_temperatures(start_time=None,end_time=None,path='/grp/hst/OTA/thermal'):
    ''' Add all temperatures for initialization of
    the table.


    Add check for start, end going beyond range of temperatures.

    Need to go back in time a bit to get all data for a given time.
    '''

    # This is inefficient, but we read in every temperature
    fnames = sorted(glob(os.path.join(path, 'thermalData*.dat')))

    assert len(fnames) > 0, 'No files matching "thermalData*.dat" found at "{}"!'.format(os.path.abspath(path))

    # Parse each file
    temp_list = []
    for f in fnames:
        temp_list.extend( _parse_temperature(f) )

    # Convert to nd array and reduced to requested time limits
    temp_array = np.asarray(temp_list).T
    earliest = temp_array[0].min()
    latest = temp_array[0].max()
    if start_time is None:
        start_time = earliest
    if end_time is None:
        end_time = latest
    assert start_time >= earliest, 'Cannot request a time earlier than {} MJD'.format(earliest)
    assert end_time <= latest, 'Cannot request a time later than {} MJD'.format(latest)
    good_idx = (temp_array[0] >= start_time) & (temp_array[0] <= end_time)

    columns = ['mjd','aftLS','trussAxial','trussDiam','aftShroud','fwdShell','lightShield']
    temperatures = np.core.records.fromarrays(temp_array[:,good_idx], dtype = [(c, float) for c in columns])
    return temperatures


def to_julian(y, m, d, H = 0, M = 0, S = 0):
    ''' Convert from year, month, day to MJD with
    optional hour, minute and second arguments.

    Written by C. Cox.
    '''

    j111 = 678576 # Julian day zero referenced from (0001,1,1) the ordinal number
    j1900 = 40587 # Julian day for Jan 1 1900
    dobj = datetime.date(y, m, d) #Create datetime object
    ordinal = dobj.toordinal()
    jul = ordinal - j111
    jul = jul + ((S/60.0 + M)/60.0 + H)/24.0 # Add fraction of day
    return jul 

