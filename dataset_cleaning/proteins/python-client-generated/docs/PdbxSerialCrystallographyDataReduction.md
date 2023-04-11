# PdbxSerialCrystallographyDataReduction

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**crystal_hits** | **int** | For experiments in which samples are provided in a  continuous stream, the total number of frames collected  in which the crystal was hit. | [optional] 
**diffrn_id** | **str** | The data item is a pointer to _diffrn.id in the DIFFRN  category. | 
**droplet_hits** | **int** | For experiments in which samples are provided in a  continuous stream, the total number of frames collected  in which a droplet was hit. | [optional] 
**frame_hits** | **int** | For experiments in which samples are provided in a  continuous stream, the total number of data frames collected  in which the sample was hit. | [optional] 
**frames_failed_index** | **int** | For experiments in which samples are provided in a  continuous stream, the total number of data frames collected  that contained a \&quot;hit\&quot; but failed to index. | [optional] 
**frames_indexed** | **int** | For experiments in which samples are provided in a  continuous stream, the total number of data frames collected  that were indexed. | [optional] 
**frames_total** | **int** | The total number of data frames collected for this  data set. | [optional] 
**lattices_indexed** | **int** | For experiments in which samples are provided in a  continuous stream, the total number of lattices indexed. | [optional] 
**lattices_merged** | **int** | For experiments in which samples are provided in a             continuous stream, the total number of crystal lattices             that were merged in the final dataset.  Can be             less than frames_indexed depending on filtering during merging or      can be more than frames_indexed if there are multiple lattices.      per frame. | [optional] 
**xfel_pulse_events** | **int** | For FEL experiments, the number of pulse events in the dataset. | [optional] 
**xfel_run_numbers** | **str** | For FEL experiments, in which data collection was performed         in batches, indicates which subset of the data collected                were used in producing this dataset. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

