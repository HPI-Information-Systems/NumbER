# PdbxNmrSampleDetails

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**contents** | **str** | A complete description of each NMR sample. Include the concentration and concentration units for each component (include buffers, etc.). For each component describe the isotopic composition, including the % labeling level, if known.  For example: 1. Uniform (random) labeling with 15N: U-15N 2. Uniform (random) labeling with 13C, 15N at known labeling    levels: U-95% 13C;U-98% 15N 3. Residue selective labeling: U-95% 15N-Thymine 4. Site specific labeling: 95% 13C-Ala18, 5. Natural abundance labeling in an otherwise uniformly labeled    biomolecule is designated by NA: U-13C; NA-K,H | [optional] 
**details** | **str** | Brief description of the sample providing additional information not captured by other items in the category. | [optional] 
**label** | **str** | A value that uniquely identifies this sample from the other samples listed in the entry. | [optional] 
**solution_id** | **str** | The name (number) of the sample. | 
**solvent_system** | **str** | The solvent system used for this sample. | [optional] 
**type** | **str** | A descriptive term for the sample that defines the general physical properties of the sample. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

