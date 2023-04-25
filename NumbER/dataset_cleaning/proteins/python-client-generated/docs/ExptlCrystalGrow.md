# ExptlCrystalGrow

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**crystal_id** | **str** | This data item is a pointer to _exptl_crystal.id in the  EXPTL_CRYSTAL category. | 
**details** | **str** | A description of special aspects of the crystal growth. | [optional] 
**method** | **str** | The method used to grow the crystals. | [optional] 
**p_h** | **float** | The pH at which the crystal was grown. If more than one pH was  employed during the crystallization process, the final pH should  be noted here and the protocol involving multiple pH values  should be described in _exptl_crystal_grow.details. | [optional] 
**pdbx_details** | **str** | Text description of crystal growth procedure. | [optional] 
**pdbx_p_h_range** | **str** | The range of pH values at which the crystal was grown.   Used when  a point estimate of pH is not appropriate. | [optional] 
**temp** | **float** | The temperature in kelvins at which the crystal was grown.  If more than one temperature was employed during the  crystallization process, the final temperature should be noted  here and the protocol  involving multiple temperatures should be  described in _exptl_crystal_grow.details. | [optional] 
**temp_details** | **str** | A description of special aspects of temperature control during  crystal growth. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

