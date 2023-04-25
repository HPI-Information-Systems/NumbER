# EmDiffractionStats

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**details** | **str** | Any addition details about the structure factor measurements | [optional] 
**fourier_space_coverage** | **float** | Completeness of the structure factor data within the defined space group  at the reported resolution (percent). | [optional] 
**high_resolution** | **float** | High resolution limit of the structure factor data, in angstroms | [optional] 
**id** | **str** | PRIMARY KEY | 
**image_processing_id** | **str** | Pointer to _em_image_processing.id | [optional] 
**num_intensities_measured** | **int** | Total number of diffraction intensities measured (before averaging) | [optional] 
**num_structure_factors** | **int** | Number of structure factors obtained (merged amplitudes + phases) | [optional] 
**overall_phase_error** | **float** | Overall phase error in degrees | [optional] 
**overall_phase_residual** | **float** | Overall phase residual in degrees | [optional] 
**phase_error_rejection_criteria** | **str** | Criteria used to reject phases | [optional] 
**r_merge** | **float** | Rmerge value (percent) | [optional] 
**r_sym** | **float** | Rsym value (percent) | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

