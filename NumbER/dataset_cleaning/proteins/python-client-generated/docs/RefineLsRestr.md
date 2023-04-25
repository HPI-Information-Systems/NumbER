# RefineLsRestr

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dev_ideal** | **float** | For the given parameter type, the root-mean-square deviation  between the ideal values used as restraints in the least-squares  refinement and the values obtained by refinement. For instance,  bond distances may deviate by 0.018 \\%A (r.m.s.) from ideal  values in the current model. | [optional] 
**dev_ideal_target** | **float** | For the given parameter type, the target root-mean-square  deviation between the ideal values used as restraints in the  least-squares refinement and the values obtained by refinement. | [optional] 
**number** | **int** | The number of parameters of this type subjected to restraint in  least-squares refinement. | [optional] 
**pdbx_refine_id** | **str** | This data item uniquely identifies a refinement within an entry.  _refine_ls_restr.pdbx_refine_id can be used to distinguish the results  of joint refinements. | 
**pdbx_restraint_function** | **str** | The functional form of the restraint function used in the least-squares  refinement. | [optional] 
**type** | **str** | The type of the parameter being restrained.  Explicit sets of data values are provided for the programs  PROTIN/PROLSQ (beginning with p_) and RESTRAIN (beginning with  RESTRAIN_). As computer programs change, these data values  are given as examples, not as an enumeration list. Computer  programs that convert a data block to a refinement table will  expect the exact form of the data values given here to be used. | 
**weight** | **float** | The weighting value applied to this type of restraint in  the least-squares refinement. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

