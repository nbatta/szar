from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

####################################################################
####################################################################
### LAT CALCULATOR ###
####################################################################
####################################################################
def Simons_Observatory_V3_LA_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimate these for you
    return(np.array([27.,39.,93.,145.,225.,280.]))

def Simons_Observatory_V3_LA_beams():
    ## returns the LAC beams in arcminutes
    beam_LAC_27  = 7.4
    beam_LAC_39  = 5.1
    beam_LAC_93  = 2.2
    beam_LAC_145 = 1.4
    beam_LAC_225 = 1.0
    beam_LAC_280 = 0.9
    return(np.array([beam_LAC_27,beam_LAC_39,beam_LAC_93,beam_LAC_145,beam_LAC_225,beam_LAC_280]))

def Simons_Observatory_V3_LA_noise(sensitivity_mode,f_sky,ell_max,delta_ell,N_LF=1.,N_MF=4.,N_UHF=2.):
    ## returns noise curves, including the impact of the beam for the SO large aperture telescopes
    # sensitivity_mode
    #     0: threshold, 
    #     1: baseline, 
    #     2: goal
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the computation of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## LARGE APERTURE
    # configuration
    # ensure valid parameter choices
    assert( sensitivity_mode == 0 or sensitivity_mode == 1 or sensitivity_mode == 2)
    assert( f_sky > 0. and f_sky <= 1.)
    assert( ell_max <= 2e4 )
    assert( delta_ell >= 1 )
    # ensure total is 7
    if (N_LF + N_MF + N_UHF) != 7:
        print("WARNING! You requested:",N_LF + N_MF + N_UHF, "optics tubes while V3 includes budget for 7")
    NTubes_LF  = N_LF  #default = 1
    NTubes_MF  = N_MF  #default = 4.
    NTubes_UHF = N_UHF #default = 2.
    # sensitivity in uK*sqrt(s)
    # set noise to irrelevantly high value when NTubes=0
    # note that default noise levels are for 1-4-2 tube configuration
    if (NTubes_LF == 0.):
        S_LA_27 = 1.e9*np.ones(3)
        S_LA_39 = 1.e9*np.ones(3)
    else:
        S_LA_27  = np.array([61.,48.,35.]) * np.sqrt(1./NTubes_LF)  ## converting these to per tube sensitivities
        S_LA_39  = np.array([30.,24.,18.]) * np.sqrt(1./NTubes_LF)
    if (NTubes_MF == 0.):
        S_LA_93 = 1.e9*np.ones(3)
        S_LA_145 = 1.e9*np.ones(3)
    else:
        S_LA_93  = np.array([6.5,5.4,3.9]) * np.sqrt(4./NTubes_MF) 
        S_LA_145 = np.array([8.1,6.7,4.2]) * np.sqrt(4./NTubes_MF) 
    if (NTubes_UHF == 0.):
        S_LA_225 = 1.e9*np.ones(3)
        S_LA_280 = 1.e9*np.ones(3)
    else:
        S_LA_225 = np.array([17.,15.,10.]) * np.sqrt(2./NTubes_UHF) 
        S_LA_280 = np.array([42.,36.,25.]) * np.sqrt(2./NTubes_UHF)
    # 1/f pol:  see http://simonsobservatory.wikidot.com/review-of-hwp-large-aperture-2017-10-04
    f_knee_pol_LA_27 = 700.
    f_knee_pol_LA_39 = 700.
    f_knee_pol_LA_93 = 700.
    f_knee_pol_LA_145 = 700.
    f_knee_pol_LA_225 = 700.
    f_knee_pol_LA_280 = 700.
    alpha_pol = -1.4
    # atmospheric 1/f temp from Matthew H.'s model
    C_27  =    200.
    C_39  =    7.7
    C_93  =   1800.
    C_145 =  12000.
    C_225 =  2.66015359e+05 #68000.
    C_280 = 1.56040514e+06 #124000. 
    alpha_temp = -3.5
    
    ####################################################################
    ## calculate the survey area and time
    survey_time = 5. #years
    t = survey_time * 365.25 * 24. * 3600.    ## convert years to seconds
    t = t * 0.2   ## retention after observing efficiency and cuts
    t = t * 0.85  ## a kludge for the noise non-uniformity of the map edges
    A_SR = 4. * np.pi * f_sky  ## sky areas in Steradians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    print("sky area: ", A_deg, "degrees^2")
    
    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2,ell_max,delta_ell)
    
    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_LA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_LA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_LA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_LA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_LA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_LA_280[sensitivity_mode] / np.sqrt(t)
    
    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    MN_T_280 = W_T_280 * np.sqrt(A_arcmin)
    Map_white_noise_levels= np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    print("white noise level: ",Map_white_noise_levels ,"[uK-arcmin]")
    
    ## calculate the atmospheric contribution for T (based on Matthew's model)
    # the 2*NTube factor comes from Matthew H.'s email on 1-25-18
    ell_pivot = 1000.
    # handle cases where there are zero tubes of some kind
    if (NTubes_LF == 0.):
        AN_T_27 = 0. #irrelevantly large noise already set above
        AN_T_39 = 0.
    else:
        AN_T_27  = C_27  * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_LF) 
        AN_T_39  = C_39  * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_LF)
    if (NTubes_MF == 0.):
        AN_T_93 = 0.
        AN_T_145 = 0.
    else:
        AN_T_93  = C_93  * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_MF)
        AN_T_145 = C_145 * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_MF)
    if (NTubes_UHF == 0.):
        AN_T_225 = 0.
        AN_T_280 = 0.
    else:
        AN_T_225 = C_225 * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_UHF)
        AN_T_280 = C_280 * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_UHF)
    # include cross-frequency correlations in the atmosphere
    # Matthew H.: the most well-motivated model for atmospheric correlation between bands is as follows:
    #   - the atmospheric noise is 100% correlated between bands in a single optics tube.
    #   - the atmospheric noise is not correlated, at least for ell>400, between adjacent optics tubes.
    # use correlation coefficient of r=0.9 within each dichroic pair and 0 otherwise
    r_atm = 0.9
    AN_T_27x39 = r_atm * np.sqrt(AN_T_27 * AN_T_39)
    AN_T_93x145 = r_atm * np.sqrt(AN_T_93 * AN_T_145)
    AN_T_225x280 = r_atm * np.sqrt(AN_T_225 * AN_T_280)

    ## calculate N(ell)
    N_ell_T_27   = (W_T_27**2. * A_SR) + AN_T_27
    N_ell_T_39   = (W_T_39**2. * A_SR) + AN_T_39
    N_ell_T_93   = (W_T_93**2. * A_SR) + AN_T_93
    N_ell_T_145  = (W_T_145**2. * A_SR) + AN_T_145
    N_ell_T_225  = (W_T_225**2. * A_SR) + AN_T_225
    N_ell_T_280  = (W_T_280**2. * A_SR) + AN_T_280
    # include cross-correlations due to atmospheric noise
    N_ell_T_27x39 = AN_T_27x39 
    N_ell_T_93x145 = AN_T_93x145
    N_ell_T_225x280 = AN_T_225x280

    ## include the impact of the beam
    LA_beams = Simons_Observatory_V3_LA_beams() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
                                ## lac beams as a sigma expressed in radians
    N_ell_T_27  *= np.exp( ell*(ell+1)* LA_beams[0]**2. )
    N_ell_T_39  *= np.exp( ell*(ell+1)* LA_beams[1]**2. )
    N_ell_T_93  *= np.exp( ell*(ell+1)* LA_beams[2]**2. )
    N_ell_T_145 *= np.exp( ell*(ell+1)* LA_beams[3]**2. )
    N_ell_T_225 *= np.exp( ell*(ell+1)* LA_beams[4]**2. )
    N_ell_T_280 *= np.exp( ell*(ell+1)* LA_beams[5]**2. )
    N_ell_T_27x39 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[0]**2. + LA_beams[1]**2.) )
    N_ell_T_93x145 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[2]**2. + LA_beams[3]**2.) )
    N_ell_T_225x280 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[4]**2. + LA_beams[5]**2.) )
    
    ## make an array of noise curves for T
    # include cross-correlations due to atmospheric noise
    N_ell_T_LA = np.array([N_ell_T_27,N_ell_T_39,N_ell_T_93,N_ell_T_145,N_ell_T_225,N_ell_T_280,N_ell_T_27x39,N_ell_T_93x145,N_ell_T_225x280])
    
    ####################################################################
    ###   CALCULATE N(ell) for Polarization
     ## calculate the astmospheric contribution for P
    AN_P_27  = (ell / f_knee_pol_LA_27 )**alpha_pol + 1.  
    AN_P_39  = (ell / f_knee_pol_LA_39 )**alpha_pol + 1. 
    AN_P_93  = (ell / f_knee_pol_LA_93 )**alpha_pol + 1.   
    AN_P_145 = (ell / f_knee_pol_LA_145)**alpha_pol + 1.   
    AN_P_225 = (ell / f_knee_pol_LA_225)**alpha_pol + 1.   
    AN_P_280 = (ell / f_knee_pol_LA_280)**alpha_pol + 1.

    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2. * A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2. * A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2. * A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2. * A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2. * A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2. * A_SR * AN_P_280
    # include cross-correlations due to atmospheric noise
    # different approach than for T -- need to subtract off the white noise part to get the purely atmospheric part
    N_ell_P_27_atm = (W_T_27  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_27 )**alpha_pol
    N_ell_P_39_atm = (W_T_39  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_39 )**alpha_pol
    N_ell_P_93_atm = (W_T_93  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_93 )**alpha_pol
    N_ell_P_145_atm = (W_T_145  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_145 )**alpha_pol
    N_ell_P_225_atm = (W_T_225  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_225 )**alpha_pol
    N_ell_P_280_atm = (W_T_280  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_280 )**alpha_pol
    N_ell_P_27x39 = r_atm * np.sqrt(N_ell_P_27_atm * N_ell_P_39_atm)
    N_ell_P_93x145 = r_atm * np.sqrt(N_ell_P_93_atm * N_ell_P_145_atm)
    N_ell_P_225x280 = r_atm * np.sqrt(N_ell_P_225_atm * N_ell_P_280_atm)
        
    ## include the imapct of the beam
    N_ell_P_27  *= np.exp( ell*(ell+1)* LA_beams[0]**2 )
    N_ell_P_39  *= np.exp( ell*(ell+1)* LA_beams[1]**2 )
    N_ell_P_93  *= np.exp( ell*(ell+1)* LA_beams[2]**2 )
    N_ell_P_145 *= np.exp( ell*(ell+1)* LA_beams[3]**2 )
    N_ell_P_225 *= np.exp( ell*(ell+1)* LA_beams[4]**2 )
    N_ell_P_280 *= np.exp( ell*(ell+1)* LA_beams[5]**2 )
    N_ell_P_27x39 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[0]**2. + LA_beams[1]**2.) )
    N_ell_P_93x145 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[2]**2. + LA_beams[3]**2.) )
    N_ell_P_225x280 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[4]**2. + LA_beams[5]**2.) )
    
    ## make an array of noise curves for P
    N_ell_P_LA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280,N_ell_P_27x39,N_ell_P_93x145,N_ell_P_225x280])
    
    ####################################################################
    return(ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels)

####################################################################
####################################################################
### SAC CALCULATOR ###
####################################################################
####################################################################
def Simons_Observatory_V3_SA_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimate these for you
    return(np.array([27.,39.,93.,145.,225.,280.]))

def Simons_Observatory_V3_SA_beams():
    ## returns the SAC beams in arcminutes
    beam_SAC_27 = 91.
    beam_SAC_39 = 63.
    beam_SAC_93 = 30.
    beam_SAC_145 = 17.
    beam_SAC_225 = 11.
    beam_SAC_280 = 9.
    return(np.array([beam_SAC_27,beam_SAC_39,beam_SAC_93,beam_SAC_145,beam_SAC_225,beam_SAC_280]))

def Simons_Observatory_V3_SA_noise(sensitivity_mode,one_over_f_mode,SAC_yrs_LF,f_sky,ell_max,delta_ell):
    ## retuns noise curves, including the impact of the beam for the SO small aperture telescopes
    ## noise curves are polarization only
    # sensitivity_mode
    #     0: threshold, 
    #     1: baseline, 
    #     2: goal
    # one_over_f_mode
    #     0: pessimistic
    #     1: optimistic
    # SAC_yrs_LF: 0,1,2,3,4,5:  number of years where an LF is deployed on SAC
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the computation of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## SMALL APERTURE
    # ensure valid parameter choices
    assert( sensitivity_mode == 0 or sensitivity_mode == 1 or sensitivity_mode == 2)
    assert( one_over_f_mode == 0 or one_over_f_mode == 1)
    assert( SAC_yrs_LF <= 5) #N.B. SAC_yrs_LF can be negative
    assert( f_sky > 0. and f_sky <= 1.)
    assert( ell_max <= 2e4 )
    assert( delta_ell >= 1 )
    # configuration
    if (SAC_yrs_LF > 0):
        NTubes_LF  = SAC_yrs_LF/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2 - SAC_yrs_LF/5.
    else:
        NTubes_LF  = np.fabs(SAC_yrs_LF)/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2 
    NTubes_UHF = 1.
    # sensitivity
    # N.B. divide-by-zero will occur if NTubes = 0
    # handle with assert() since it's highly unlikely we want any configurations without >= 1 of each tube type
    assert( NTubes_LF > 0. )
    assert( NTubes_MF > 0. )
    assert( NTubes_UHF > 0.)
    S_SA_27  = np.array([32,21,15])    * np.sqrt(1./NTubes_LF)
    S_SA_39  = np.array([17,13,10])    * np.sqrt(1./NTubes_LF)
    S_SA_93  = np.array([4.6,3.4,2.4]) * np.sqrt(2./(NTubes_MF))
    S_SA_145 = np.array([5.5,4.3,2.7]) * np.sqrt(2./(NTubes_MF))
    S_SA_225 = np.array([11,8.6,5.7])  * np.sqrt(1./NTubes_UHF)
    S_SA_280 = np.array([26,22,14])    * np.sqrt(1./NTubes_UHF)
    # 1/f pol:  see http://simonsobservatory.wikidot.com/review-of-hwp-large-aperture-2017-10-04
    f_knee_pol_SA_27  = np.array([30.,15.])
    f_knee_pol_SA_39  = np.array([30.,15.])  ## from QUIET
    f_knee_pol_SA_93  = np.array([50.,25.])
    f_knee_pol_SA_145 = np.array([50.,25.])  ## from ABS, improving possible by scanning faster
    f_knee_pol_SA_225 = np.array([70.,35.])
    f_knee_pol_SA_280 = np.array([100.,40.])
    alpha_pol =np.array([-2.4,-2.4,-2.5,-3,-3,-3])  ## roughly consistent with Yuji's table, but extrapolated

    ####################################################################
    ## calculate the survey area and time
    t = 5* 365. * 24. * 3600    ## five years in seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    t = t* 0.85  ## a kluge for the noise non-uniformity of the map edges
    A_SR = 4 * np.pi * f_sky  ## sky areas in Steradians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    print("sky area: ", A_deg, "degrees^2")
    print("when generating realizations from a hits map, the total integration time should be 1/0.85 longer")
    print("since we should remove a kluge for map non-uniformity since this is included correcly in a hits map")

    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2,ell_max,delta_ell)

    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_SA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_SA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_SA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_SA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_SA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_SA_280[sensitivity_mode] / np.sqrt(t)

    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    MN_T_280 = W_T_280 * np.sqrt(A_arcmin)
    Map_white_noise_levels = np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    print("white noise level: ",Map_white_noise_levels ,"[uK-arcmin]")

    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the atmospheric contribution for P
    AN_P_27  = (ell / f_knee_pol_SA_27[one_over_f_mode] )**alpha_pol[0] + 1.  
    AN_P_39  = (ell / f_knee_pol_SA_39[one_over_f_mode] )**alpha_pol[1] + 1. 
    AN_P_93  = (ell / f_knee_pol_SA_93[one_over_f_mode] )**alpha_pol[2] + 1.   
    AN_P_145 = (ell / f_knee_pol_SA_145[one_over_f_mode])**alpha_pol[3] + 1.   
    AN_P_225 = (ell / f_knee_pol_SA_225[one_over_f_mode])**alpha_pol[4] + 1.   
    AN_P_280 = (ell / f_knee_pol_SA_280[one_over_f_mode])**alpha_pol[5] + 1.  

    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2.* A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2.* A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2.* A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2.* A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2.* A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2.* A_SR * AN_P_280

    ## include the impact of the beam
    SA_beams = Simons_Observatory_V3_SA_beams() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
                                ## SAC beams as a sigma expressed in radians
    N_ell_P_27  *= np.exp( ell*(ell+1)* SA_beams[0]**2. )
    N_ell_P_39  *= np.exp( ell*(ell+1)* SA_beams[1]**2. )
    N_ell_P_93  *= np.exp( ell*(ell+1)* SA_beams[2]**2. )
    N_ell_P_145 *= np.exp( ell*(ell+1)* SA_beams[3]**2. )
    N_ell_P_225 *= np.exp( ell*(ell+1)* SA_beams[4]**2. )
    N_ell_P_280 *= np.exp( ell*(ell+1)* SA_beams[5]**2. )

    ## make an array of noise curves for P
    N_ell_P_SA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280])

    ####################################################################
    return(ell,N_ell_P_SA,Map_white_noise_levels)

####################################################################
####################################################################
### AdvACT CALCULATOR ###
####################################################################
####################################################################
def AdvACT_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimate these for you
    return(np.array([27.,39.,93.,145.,225.]))

def AdvACT_beams():
    ## returns the AdvACT beams in arcminutes
    beam_AdvACT_27  = 7.4  ## not measured
    beam_AdvACT_39  = 5.1  ## not measured
    beam_AdvACT_93  = 2.2
    beam_AdvACT_145 = 1.4
    beam_AdvACT_225 = 1.0
    return(np.array([beam_AdvACT_27,beam_AdvACT_39,beam_AdvACT_93,beam_AdvACT_145,beam_AdvACT_225]))

def AdvACT_noise(f_sky,ell_max,delta_ell,N_LF_years=2.,N_seasons=4.):
    ## returns noise curves, including the impact of the beam for the SO large aperture telescopes
    # N_seasons
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the computation of N(ell)
    # delta_ell: the step size for computing N_ell
    # N_LF: number of years we deploy an LF tube at the expese of a MF
    ##. LF not implemented since the noise is ucertain since it hasn't yet been fielded
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## LARGE APERTURE
    # configuration
    # ensure valid parameter choices
    assert( N_seasons >= 1. and N_seasons <= 5.)
    assert( f_sky > 0.005 and f_sky <= .7)
    assert( ell_max <= 2e4 )
    assert( delta_ell >= 1 )
    # ensure total is 7
    NTube_years_LF  = N_LF_years  #default = 1
    NTube_years_MF  = 2.*N_seasons - N_LF_years 
    NTube_years_HF = N_seasons
    # sensitivity in uK*sqrt(s)
    # set noise to irrelevantly high value when NTubes=0
    # note that default noise levels are for 1-4-2 tube configuration
    if (NTube_years_LF == 0.):
        S_27 = 1.e9
        S_39 = 1.e9
    else:
        S_27  = 45 / np.sqrt(NTube_years_LF)  ## converting these to per tube sensitivities
        S_39  = 26. / np.sqrt(NTube_years_LF)
    if (NTube_years_MF == 0.):
        S_93 = 1.e9
        S_145 = 1.e9
    else:
        S_93  = 10.67 / np.sqrt(NTube_years_MF) 
        S_145 = 12.16 / np.sqrt(NTube_years_MF + NTube_years_HF)
                ### add a thing in for the HF tube's contribution to the 145 GHz sensitivity
    if (NTube_years_HF == 0.):
        S_225 = 1.e9
    else:
        S_225 = 24. / np.sqrt(NTube_years_HF) 
    # 1/f pol:  see http://simonsobservatory.wikidot.com/review-of-hwp-large-aperture-2017-10-04
    f_knee_pol_27 = 700.
    f_knee_pol_39 = 700.
    f_knee_pol_93 = 700.
    f_knee_pol_145 = 700.
    f_knee_pol_225 = 700.
    f_knee_pol_280 = 700.
    alpha_pol = -1.4
    # atmospheric 1/f temp from Matthew H.'s model
    C_27  =    200.
    C_39  =    7.7
    C_93  =   1800.
    C_145 =  12000.
    C_225 =  68000.
    C_280 = 124000. 
    alpha_temp = -3.5
    
    ####################################################################
    ## calculate the survey area and time
    survey_time = 1. #years--- given we are using "tube-years above, this should not be changed.
    t = survey_time * 365.25 * 24. * 3600.    ## convert years to seconds
    t = t * 0.2 *0.5  ## retention after observing efficiency and cuts, daytime only
    t = t * 0.85  ## a kludge for the noise non-uniformity of the map edges
    A_SR = 4. * np.pi * f_sky  ## sky areas in Steradians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    print("sky area: ", A_deg, "degrees^2")
    
    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2,ell_max,delta_ell)
    
    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_27  / np.sqrt(t)
    W_T_39  = S_39  / np.sqrt(t)
    W_T_93  = S_93  / np.sqrt(t)
    W_T_145 = S_145 / np.sqrt(t)
    W_T_225 = S_225 / np.sqrt(t)
    
    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    Map_white_noise_levels= np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225])
    print("white noise level: ",Map_white_noise_levels ,"[uK-arcmin]")
    
    ## calculate the atmospheric contribution for T (based on Matthew's model)
    # the 2*NTube factor comes from Matthew H.'s email on 1-25-18
    ell_pivot = 1000.
    # handle cases where there are zero tubes of some kind
    if (NTube_years_LF == 0.):
        AN_T_27 = 0. #irrelevantly large noise already set above
        AN_T_39 = 0.
    else:
        AN_T_27  = C_27  * (ell/ell_pivot)**alpha_temp * A_SR / t / NTube_years_LF
        AN_T_39  = C_39  * (ell/ell_pivot)**alpha_temp * A_SR / t / NTube_years_LF
    if (NTube_years_MF == 0.):
        AN_T_93 = 0.
        AN_T_145 = 0.
    else:
        AN_T_93  = C_93  * (ell/ell_pivot)**alpha_temp * A_SR / t / NTube_years_MF
        AN_T_145 = C_145 * (ell/ell_pivot)**alpha_temp * A_SR / t / (NTube_years_MF + NTube_years_HF)
    if (NTube_years_HF == 0.):
        AN_T_225 = 0.
    else:
        AN_T_225 = C_225 * (ell/ell_pivot)**alpha_temp * A_SR / t / NTube_years_HF
    # include cross-frequency correlations in the atmosphere
    # Matthew H.: the most well-motivated model for atmospheric correlation between bands is as follows:
    #   - the atmospheric noise is 100% correlated between bands in a single optics tube.
    #   - the atmospheric noise is not correlated, at least for ell>400, between adjacent optics tubes.
    # use correlation coefficient of r=0.9 within each dichroic pair and 0 otherwise
    r_atm = 0.9
    AN_T_27x39 = r_atm * np.sqrt(AN_T_27 * AN_T_39)
    AN_T_93x145 = r_atm * np.sqrt(AN_T_93 * AN_T_145)
    AN_T_145x225 = r_atm * np.sqrt(AN_T_145 * AN_T_225) * (NTube_years_HF / (NTube_years_MF+NTube_years_HF))

    ## calculate N(ell)
    N_ell_T_27   = (W_T_27**2. * A_SR) + AN_T_27
    N_ell_T_39   = (W_T_39**2. * A_SR) + AN_T_39
    N_ell_T_93   = (W_T_93**2. * A_SR) + AN_T_93
    N_ell_T_145  = (W_T_145**2. * A_SR) + AN_T_145
    N_ell_T_225  = (W_T_225**2. * A_SR) + AN_T_225
    # include cross-correlations due to atmospheric noise
    N_ell_T_27x39 = AN_T_27x39 
    N_ell_T_93x145 = AN_T_93x145
    N_ell_T_145x225 = AN_T_145x225

    ## include the impact of the beam
    LA_beams = AdvACT_beams() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
                                ## lac beams as a sigma expressed in radians
    N_ell_T_27  *= np.exp( ell*(ell+1)* LA_beams[0]**2. )
    N_ell_T_39  *= np.exp( ell*(ell+1)* LA_beams[1]**2. )
    N_ell_T_93  *= np.exp( ell*(ell+1)* LA_beams[2]**2. )
    N_ell_T_145 *= np.exp( ell*(ell+1)* LA_beams[3]**2. )
    N_ell_T_225 *= np.exp( ell*(ell+1)* LA_beams[4]**2. )
    N_ell_T_27x39 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[0]**2. + LA_beams[1]**2.) )
    N_ell_T_93x145 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[2]**2. + LA_beams[3]**2.) )
    N_ell_T_145x225 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[3]**2. + LA_beams[4]**2.) )
    
    ## make an array of noise curves for T
    # include cross-correlations due to atmospheric noise
    N_ell_T_AdvACT = np.array([N_ell_T_27,N_ell_T_39,N_ell_T_93,N_ell_T_145,N_ell_T_225,N_ell_T_27x39,N_ell_T_93x145,N_ell_T_145x225])
    
    ####################################################################
    ###   CALCULATE N(ell) for Polarization
     ## calculate the astmospheric contribution for P
    AN_P_27  = (ell / f_knee_pol_27 )**alpha_pol + 1.  
    AN_P_39  = (ell / f_knee_pol_39 )**alpha_pol + 1. 
    AN_P_93  = (ell / f_knee_pol_93 )**alpha_pol + 1.   
    AN_P_145 = (ell / f_knee_pol_145)**alpha_pol + 1.   
    AN_P_225 = (ell / f_knee_pol_225)**alpha_pol + 1.   


    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2. * A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2. * A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2. * A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2. * A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2. * A_SR * AN_P_225
    # include cross-correlations due to atmospheric noise
    # different approach than for T -- need to subtract off the white noise part to get the purely atmospheric part
    N_ell_P_27_atm = (W_T_27  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_27 )**alpha_pol
    N_ell_P_39_atm = (W_T_39  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_39 )**alpha_pol
    N_ell_P_93_atm = (W_T_93  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_93 )**alpha_pol
    N_ell_P_145_atm =(W_T_145 * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_145)**alpha_pol
    N_ell_P_225_atm =(W_T_225 * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_225)**alpha_pol
    N_ell_P_27x39 = r_atm * np.sqrt(N_ell_P_27_atm * N_ell_P_39_atm)
    N_ell_P_93x145 = r_atm * np.sqrt(N_ell_P_93_atm * N_ell_P_145_atm)
    N_ell_P_145x225 = r_atm * np.sqrt(N_ell_P_145_atm * N_ell_P_225_atm) * (NTube_years_HF / (NTube_years_MF+NTube_years_HF))
        
    ## include the imapct of the beam
    N_ell_P_27  *= np.exp( ell*(ell+1)* LA_beams[0]**2 )
    N_ell_P_39  *= np.exp( ell*(ell+1)* LA_beams[1]**2 )
    N_ell_P_93  *= np.exp( ell*(ell+1)* LA_beams[2]**2 )
    N_ell_P_145 *= np.exp( ell*(ell+1)* LA_beams[3]**2 )
    N_ell_P_225 *= np.exp( ell*(ell+1)* LA_beams[4]**2 )
    N_ell_P_27x39 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[0]**2. + LA_beams[1]**2.) )
    N_ell_P_93x145 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[2]**2. + LA_beams[3]**2.) )
    N_ell_P_145x225 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[3]**2. + LA_beams[4]**2.) )
    
    ## make an array of noise curves for P
    N_ell_P_AdvACT = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_27x39,N_ell_P_93x145,N_ell_P_145x225])
    
    ####################################################################
    return(ell, N_ell_T_AdvACT, N_ell_P_AdvACT, Map_white_noise_levels)

def main():
    ####################################################################
    ####################################################################
    ##                   demonstration of the code
    ####################################################################
    print("band centers: ", Simons_Observatory_V3_LA_bands(), "[GHz]")
    print("beam sizes: "  , Simons_Observatory_V3_LA_beams(), "[arcmin]")
    N_bands = np.size(Simons_Observatory_V3_LA_bands())
    beams = Simons_Observatory_V3_LA_beams()
    beams_sigma_rad = beams / np.sqrt(8. * np.log(2)) /60. * np.pi/180.

    ## run the code to generate noise curves
    mode=2
    fsky=0.4
    N_LF=1.
    N_MF=4.
    N_HF=0.
    N_UHF=2.
    ellmax=1e4
    ell, N_ell_LA_T,N_ell_LA_Pol,WN_levels = Simons_Observatory_V3_LA_noise(mode,fsky,ellmax,1,N_LF,N_MF,N_UHF)
    print("white noise levels: "  , WN_levels, "[uK-arcmin]")
    N_ell_V3_T_white = np.zeros((N_bands,np.int(ellmax-2)))
    for i in xrange(N_bands):
        N_ell_V3_T_white[i] = (WN_levels[i]**2. * 1./(1.*(180./np.pi)**2.*3600.) * np.exp( ell*(ell+1)* beams_sigma_rad[i]**2. ))

    ## plot the temperature noise curves
    colors=['b','r','g','m','k','y']
    plt.clf()
    i = 0
    while (i < N_bands):
        plt.loglog(ell,N_ell_LA_T[i], label=str((Simons_Observatory_V3_LA_bands())[i])+' GHz (V3)', color=colors[i], ls='-', lw=2.)
        #plt.loglog(ell,N_ell_V3_T_white[i], color=colors[i], ls='-', lw=0.5) #white noise
        i+=1
    # include correlated atmospheric noise across frequencies
    plt.loglog(ell, N_ell_LA_T[N_bands], label=r'$27 \times 39$ GHz atm.', color='orange', lw=1.5)
    plt.loglog(ell, N_ell_LA_T[N_bands+1], label=r'$93 \times 145$ GHz atm.', color='fuchsia', lw=1.5)
    plt.loglog(ell, N_ell_LA_T[N_bands+2], label=r'$225 \times 280$ GHz atm.', color='springgreen', lw=1.5)
    plt.title(r"$N(\ell$) Temperature", fontsize=18)
    plt.ylabel(r"$N(\ell$) [$\mu$K${}^2$]", fontsize=16)
    plt.xlabel(r"$\ell$", fontsize=16)
    plt.ylim(5e-7,1)
    plt.xlim(100,10000)
    plt.legend(loc='lower left', ncol=2, fontsize=8)
    plt.grid()
    plt.savefig('V3_calc_mode'+str(mode)+'_fsky'+str(fsky)+'_defaultdist_noise_LAT_T.pdf')
    plt.close()

    ## plot the polarization noise curves
    plt.clf()
    i = 0
    while (i < N_bands):
        plt.loglog(ell,N_ell_LA_Pol[i], label=str((Simons_Observatory_V3_LA_bands())[i])+' GHz (V3)', color=colors[i], ls='-', lw=2.)
        #plt.loglog(ell,N_ell_V3_T_white[i], color=colors[i], ls='-', lw=0.5) #white noise
        i+=1
    # include correlated atmospheric noise across frequencies
    plt.loglog(ell, N_ell_LA_Pol[N_bands], label=r'$27 \times 39$ GHz atm.', color='orange', lw=1.5)
    plt.loglog(ell, N_ell_LA_Pol[N_bands+1], label=r'$93 \times 145$ GHz atm.', color='fuchsia', lw=1.5)
    plt.loglog(ell, N_ell_LA_Pol[N_bands+2], label=r'$225 \times 280$ GHz atm.', color='springgreen', lw=1.5)
    plt.title(r"$N(\ell$) Polarization", fontsize=18)
    plt.ylabel(r"$N(\ell$) [$\mu$K${}^2$]", fontsize=16)
    plt.xlabel(r"$\ell$", fontsize=16)
    plt.ylim(5e-7,1)
    plt.xlim(100,10000)
    plt.legend(loc='upper left', ncol=2, fontsize=9)
    plt.grid()
    plt.savefig('V3_calc_mode'+str(mode)+'_fsky'+str(fsky)+'_defaultdist_noise_LAT_P.pdf')
    plt.close()
    ####################################################################
    ####################################################################


    ####################################################################
    ####################################################################
    ##                   demonstration of the code
    ####################################################################
    print("band centers: ", Simons_Observatory_V3_SA_bands(), "[GHz]")
    print("beam sizes: "  , Simons_Observatory_V3_SA_beams(), "[arcminute]")

    ## run the code to generate noise curves
    fsky_SAC = 0.1
    sens_mode = 2
    one_over_f_mode_SAC = 0
    SAC_yrs_LF = 2
    ell, N_ell_SA_Pol,WN_levels = Simons_Observatory_V3_SA_noise(sens_mode,one_over_f_mode_SAC,SAC_yrs_LF,fsky_SAC,500,1)

    ## plot the polarization noise curves
    N_bands = np.size(Simons_Observatory_V3_LA_bands())
    i = 0
    while (i < N_bands):
        plt.loglog(ell,N_ell_SA_Pol[i], label=str((Simons_Observatory_V3_SA_bands())[i])+' GHz (V3)', color=colors[i], ls='-', lw=2.)
        i+=1
    plt.title(r"$N(\ell$) Polarization", fontsize=18)
    plt.ylabel(r"$N(\ell$) [$\mu$K${}^2$]", fontsize=16)
    plt.xlabel(r"$\ell$", fontsize=16)
    plt.ylim(1e-6,0.01)
    plt.xlim(10,500)
    plt.legend(loc='lower left', ncol=2, fontsize=8)
    plt.grid()
    plt.savefig('V3_calc_mode'+str(sens_mode)+'-'+str(one_over_f_mode_SAC)+'_SACyrsLF'+str(SAC_yrs_LF)+'_fsky'+str(fsky)+'_defaultdist_noise_SAC_P_ASRfix.pdf')
    plt.close()


