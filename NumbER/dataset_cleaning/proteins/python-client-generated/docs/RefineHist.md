# RefineHist

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**cycle_id** | **str** | The value of _refine_hist.cycle_id must uniquely identify a  record in the REFINE_HIST list.   Note that this item need not be a number; it can be any unique  identifier. | 
**d_res_high** | **float** | The lowest value for the interplanar spacings for the  reflection data for this cycle of refinement. This is called  the highest resolution. | [optional] 
**d_res_low** | **float** | The highest value for the interplanar spacings for the  reflection data for this cycle of refinement. This is  called the lowest resolution. | [optional] 
**number_atoms_solvent** | **int** | The number of solvent atoms that were included in the model at  this cycle of the refinement. | [optional] 
**number_atoms_total** | **int** | The total number of atoms that were included in the model at  this cycle of the refinement. | [optional] 
**pdbx_b_iso_mean_ligand** | **float** | Mean isotropic B-value for ligand molecules included in refinement. | [optional] 
**pdbx_b_iso_mean_solvent** | **float** | Mean isotropic B-value for solvent molecules included in refinement. | [optional] 
**pdbx_number_atoms_ligand** | **int** | Number of ligand atoms included in refinement | [optional] 
**pdbx_number_atoms_nucleic_acid** | **int** | Number of nucleic atoms included in refinement | [optional] 
**pdbx_number_atoms_protein** | **int** | Number of protein atoms included in refinement | [optional] 
**pdbx_number_residues_total** | **int** | Total number of polymer residues included in refinement. | [optional] 
**pdbx_refine_id** | **str** | This data item uniquely identifies a refinement within an entry.  _refine_hist.pdbx_refine_id can be used to distinguish the results  of joint refinements. | 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

