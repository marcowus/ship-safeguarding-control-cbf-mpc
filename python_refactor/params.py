import numpy as np

def get_params():
    """
    Returns a dictionary of parameters for the naval vessel.
    """
    params = {}

    params['rho_water'] = 1014.0
    params['rho_air']   = 1.225
    params['g']         = 9.81
    params['deg2rad']   = np.pi/180
    params['rad2deg']   = 180/np.pi
    params['ms2kt']     = 3600/1852
    params['kt2ms']     = 1852/3600
    params['RPM2rads']  = 2*np.pi/60
    params['rads2RPM']  = 60/(2*np.pi)
    params['HP2W']      = 745.700

    params['sp']        = 1.5
    params['A']         = 1.5
    params['ar']        = 3
    params['dCL']       = 0.054
    params['stall']     = 23

    params['Lpp']       = 51.5
    params['B']         = 8.6
    params['D']         = 2.3

    params['disp']      = 357.0
    params['m']         = params['disp'] * params['rho_water']
    params['Izz']       = 47.934e6
    params['Ixx']       = 2.3763e6
    params['U_nom']     = 8.0
    params['KM']        = 4.47
    params['KB']        = 1.53
    params['gm']        = 1.1
    params['bm']        = params['KM'] - params['KB']
    params['LCG']       = 20.41
    params['VCG']       = 3.36
    params['xG']        = -3.38
    params['zG']        = -(params['VCG'] - params['D'])
    params['m_xg']      = params['m'] * params['xG']
    params['m_zg']      = params['m'] * params['zG']
    params['Dp']        = 1.6

    params['Xudot']     = -17400.0
    params['Xuau']      = -1.96e3
    params['Xvr']       = 0.33 * params['m']

    params['Yvdot']     = -393000
    params['Ypdot']     = -296000
    params['Yrdot']     = -1400000
    params['Yauv']      = -11800
    params['Yur']       = 131000
    params['Yvav']      = -3700
    params['Yrar']      = 0
    params['Yvar']      = -794000
    params['Yrav']      = -182000
    params['Ybauv']     = 10800
    params['Ybaur']     = 251000
    params['Ybuu']      = -74

    params['Kvdot']     = 296000
    params['Kpdot']     = -774000
    params['Krdot']     = 0
    params['Kauv']      = 9260
    params['Kur']       = -102000
    params['Kvav']      = 29300
    params['Krar']      = 0
    params['Kvar']      = 621000
    params['Krav']      = 142000
    params['Kbauv']     = -8400
    params['Kbaur']     = -196000
    params['Kbuu']      = -1180
    params['Kaup']      = -15500
    params['Kpap']      = -416000
    params['Kp']        = -500000
    params['Kb']        = 0.776 * params['m'] * params['g']
    params['Kbbb']      = -0.325 * params['m'] * params['g']

    params['Nvdot']     = 538000
    params['Npdot']     = 0
    params['Nrdot']     = -38.7e6
    params['Nauv']      = -92000
    params['Naur']      = -4710000
    params['Nvav']      = 0
    params['Nrar']      = -202000000
    params['Nvar']      = 0
    params['Nrav']      = -15600000
    params['Nbauv']     = -214000
    params['Nbuar']     = -4980000
    params['Nbuau']     = -8000

    return params
