# RcsbNonpolymerInstanceValidationScore

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**rscc** | **float** | The real space correlation coefficient (RSCC) for the non-polymer entity instance. | [optional] 
**rsr** | **float** | The real space R-value (RSR) for the non-polymer entity instance. | [optional] 
**alt_id** | **str** | Alternate conformer identifier for the non-polymer entity instance. | [optional] 
**average_occupancy** | **float** | The average heavy atom occupancy for coordinate records for the non-polymer entity instance. | [optional] 
**completeness** | **float** | The reported fraction of atomic coordinate records for the non-polymer entity instance. | [optional] 
**intermolecular_clashes** | **int** | The number of intermolecular MolProbity clashes cacluated for reported atomic coordinate records. | [optional] 
**is_best_instance** | **str** | This molecular instance is ranked as the best quality instance of this nonpolymer entity. | [optional] 
**is_subject_of_investigation** | **str** | This molecular entity is identified as the subject of the current study. | [optional] 
**is_subject_of_investigation_provenance** | **str** | The provenance for the selection of the molecular entity identified as the subject of the current study. | [optional] 
**mogul_angle_outliers** | **int** | Number of bond angle outliers obtained from a CCDC Mogul survey of bond angles  in the CSD small    molecule crystal structure database. Outliers are defined as bond angles that have a Z-score    less than -2 or greater than 2. | [optional] 
**mogul_angles_rmsz** | **float** | The root-mean-square value of the Z-scores of bond angles for the residue in degrees obtained from a CCDC Mogul survey of bond angles in the CSD small molecule crystal structure database. | [optional] 
**mogul_bond_outliers** | **int** | Number of bond distance outliers obtained from a CCDC Mogul survey of bond lengths in the CSD small    molecule crystal structure database.  Outliers are defined as bond distances that have a Z-score    less than -2 or greater than 2. | [optional] 
**mogul_bonds_rmsz** | **float** | The root-mean-square value of the Z-scores of bond lengths for the residue in Angstroms obtained from a CCDC Mogul survey of bond lengths in the CSD small molecule crystal structure database. | [optional] 
**ranking_model_fit** | **float** | The ranking of the model fit score component. | [optional] 
**ranking_model_geometry** | **float** | The ranking of the model geometry score component. | [optional] 
**score_model_fit** | **float** | The value of the model fit score component. | [optional] 
**score_model_geometry** | **float** | The value of the model geometry score component. | [optional] 
**stereo_outliers** | **int** | Number of stereochemical/chirality errors. | [optional] 
**type** | **str** | Score type. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

