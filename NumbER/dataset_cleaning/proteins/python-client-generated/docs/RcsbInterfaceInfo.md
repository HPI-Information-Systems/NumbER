# RcsbInterfaceInfo

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**polymer_composition** | **str** |  | [optional] 
**interface_character** | **str** |  | [optional] 
**interface_area** | **float** | Total interface buried surface area | [optional] 
**self_jaccard_contact_score** | **float** | The Jaccard score (intersection over union) of interface contacts in homomeric interfaces, comparing contact sets left-right vs right-left. High values indicate isologous (symmetric) interfaces, with value&#x3D;1 if perfectly symmetric (e.g. crystallographic symmetry) | [optional] 
**num_interface_residues** | **int** | Number of interface residues, defined as those with burial fraction &gt; 0 | [optional] 
**num_core_interface_residues** | **int** | Number of core interface residues, defined as those that bury &gt;90% accessible surface area with respect to the unbound state | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

