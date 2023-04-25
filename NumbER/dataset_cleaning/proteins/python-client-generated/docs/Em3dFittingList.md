# Em3dFittingList

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**_3d_fitting_id** | **str** | The value of _em_3d_fitting_list.3d_fitting_id is a pointer  to  _em_3d_fitting.id in the 3d_fitting category | 
**details** | **str** | Details about the model used in fitting. | [optional] 
**id** | **str** | PRIMARY KEY | 
**pdb_chain_id** | **str** | The ID of the biopolymer chain used for fitting, e.g., A.  Please note that only one chain can be specified per instance.  If all chains of a particular structure have been used for fitting, this field can be left blank. | [optional] 
**pdb_chain_residue_range** | **str** | Residue range for the identified chain. | [optional] 
**pdb_entry_id** | **str** | The PDB code for the entry used in fitting. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

