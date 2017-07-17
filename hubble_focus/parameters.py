from collections import OrderedDict
from copy import deepcopy
import json
import pkg_resources

class InputParams(object):
    '''
    Set up, store, modify, print, and write out input .in files
    '''

    def __init__(self, camera_mode, nfiles=1, filepath=None):
        super().__setattr__( '_odict',  OrderedDict()  )

        defaults = get_input_defaults(filepath)
        
        assert camera_mode in defaults.keys(), \
            'Camera and Chip not recognized!'
        paramlist = deepcopy(defaults[camera_mode])
        
        if nfiles > 1:
            # When nfiles > 1, remove Focus parameter
            # and replace with focus1, focus2, etc. parameters
            # Duplicate background and x- and y-tilt parameters.
            multi_params = defaults['multi_params']

            paramlist[7] = multi_params[0]
            
            for n in range(2,nfiles+1):
                #n-th focus
                focus_n = deepcopy(multi_params[1])
                focus_n[1].strformat = focus_n[1].strformat.format('{}',n,'{: .4f}')
                paramlist.insert(6 + n, ('focus{}'.format(n), focus_n[1]) )
                
                #n-th background
                bg_n = deepcopy(multi_params[2])
                bg_n[1].strformat = bg_n[1].strformat.format('{}',n,'{: .4f}')
                paramlist.insert(18 + 2*n, ('background{}'.format(n), bg_n[1]) )
                
                #n-th tip/tilt
                tip_n = deepcopy(multi_params[3])
                tilt_n = deepcopy(multi_params[4])
                tip_n[1].strformat = tip_n[1].strformat.format('{}',n,'{: .4f}')
                tilt_n[1].strformat = tilt_n[1].strformat.format('{}',n,'{: .4f}')
                paramlist.insert(17 + 3*n + n, ('xtilt{}'.format(n), tip_n[1]) )
                paramlist.insert(18 + 3*n + n, ('ytilt{}'.format(n), tilt_n[1]) )
                
        odict = OrderedDict(paramlist)
        self._odict.update(odict)
        
    @property
    def as_string(self):
        return '\n'.join([par.as_string for par in self._odict.values()])
    
    def to_file(self, outname):
        with open(outname,'w+') as f:
            f.write(self.as_string)
        
    def __getattr__(self, key):
        odict = super().__getattribute__('_odict')
        if key in odict:
            return odict[key]
        return super().__getattribute__(key)

    def __setattr__(self, key, val):
        self._odict[key] = val

    @property
    def __dict__(self):
        return self._odict
    
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state): # Support copy.copy
        super().__setattr__( '_odict', OrderedDict() )
        self.__dict__.update( state )

class Parameter(object):
    def __init__(self,strformat,value,fit=None):
        self.strformat = strformat
        self.value = value
        self.fit = fit
    
    @property
    def as_string(self):
        if self.fit is not None:
            return self.strformat.format(self.fit,self.value)
        else:
            return self.strformat.format(self.value)

def get_input_defaults(filepath=None):
    '''
    Given a path to a JSON file with the default input parameters,
    pass back a dictionary of the default parameters(as Parameter objects).
    '''
    if filepath is None:
        filepath = pkg_resources.resource_filename(__name__,'focus_templates.json')
    with open(filepath) as f:
        templates = json.load(f)

    #get keys for each instrument. then swap out focus and add things for multi mode
    uvis_keys = ['dimensions','wavelength','fitting_method','merit_function_power','merit_wing_damping',
                 'camera_mode','zernike_type','focus','xcoma','ycoma','xastig','yastig','spherical',
                 'xclover','yclover','xspherical_astig','yspherical_astig','xashtray','yashtray','fifth_spherical',
                 'background1','xtilt1','ytilt1','blur']
    acs_keys = deepcopy(uvis_keys)
    acs_keys.insert(-1,'spider_rotation')
    multi_keys = ['focus1','focusN','backgroundN','xtiltN','ytiltN']

    wfc3uvis1_defaults = list(zip(uvis_keys, [ Parameter(*args) for args in templates['wfc3uvis1'] ]))
    wfc3uvis2_defaults = list(zip(uvis_keys, [ Parameter(*args) for args in templates['wfc3uvis2'] ]))
    acswfc1_defaults = list(zip(acs_keys, [ Parameter(*args) for args in templates['acswfc1'] ]))
    acswfc2_defaults = list(zip(acs_keys, [ Parameter(*args) for args in templates['acswfc2'] ]))
    multi_params = list(zip(multi_keys, [ Parameter(*args) for args in templates['multi_params'] ]))

    return {'wfc3uvis1' : wfc3uvis1_defaults,
            'wfc3uvis2' : wfc3uvis2_defaults,
            'acswfc1' : acswfc1_defaults,
            'acswfc2' : acswfc2_defaults,
            'multi_params' : multi_params}