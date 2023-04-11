# PdbxReflnsTwin

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**crystal_id** | **str** | The crystal identifier.  A reference to  _exptl_crystal.id in category EXPTL_CRYSTAL. | 
**diffrn_id** | **str** | The diffraction data set identifier.  A reference to  _diffrn.id in category DIFFRN. | 
**domain_id** | **str** | An identifier for the twin domain. | [optional] 
**fraction** | **float** | The twin fraction or twin factor represents a quantitative parameter for the crystal twinning.  The value 0 represents no twinning, &lt; 0.5 partial twinning,  &#x3D; 0.5 for perfect twinning. | [optional] 
**operator** | **str** | The possible merohedral or hemihedral twinning operators for different point groups are:  True point group   Twin operation   hkl related to 3                       2 along a,b             h,-h-k,-l                         2 along a*,b*           h+k,-k,-l                         2 along c               -h,-k,l 4                       2 along a,b,a*,b*       h,-k,-l 6                       2 along a,b,a*,b*       h,-h-k,-l 321                     2 along a*,b*,c         -h,-k,l 312                     2 along a,b,c           -h,-k,l 23                      4 along a,b,c            k,-h,l  References:  Yeates, T.O. (1997) Methods in Enzymology 276, 344-358. Detecting and  Overcoming Crystal Twinning.   and information from the following on-line sites:     CNS site http://cns.csb.yale.edu/v1.1/    CCP4 site http://www.ccp4.ac.uk/dist/html/detwin.html    SHELX site http://shelx.uni-ac.gwdg.de/~rherbst/twin.html | 
**type** | **str** | There are two types of twinning: merohedral or hemihedral                                  non-merohedral or epitaxial  For merohedral twinning the diffraction patterns from the different domains are completely superimposable.   Hemihedral twinning is a special case of merohedral twinning. It only involves two distinct domains.  Pseudo-merohedral twinning is a subclass merohedral twinning in which lattice is coincidentally superimposable.  In the case of non-merohedral or epitaxial twinning  the reciprocal lattices do not superimpose exactly. In this case the  diffraction pattern consists of two (or more) interpenetrating lattices, which can in principle be separated. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

