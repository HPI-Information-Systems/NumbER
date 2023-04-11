# Em3dReconstruction

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**actual_pixel_size** | **float** | The actual pixel size of the projection set of images in Angstroms. | [optional] 
**algorithm** | **str** | The reconstruction algorithm/technique used to generate the map. | [optional] 
**details** | **str** | Any additional details used in the 3d reconstruction. | [optional] 
**id** | **str** | PRIMARY KEY | 
**image_processing_id** | **str** | Foreign key to the EM_IMAGE_PROCESSING category | 
**magnification_calibration** | **str** | The magnification calibration method for the 3d reconstruction. | [optional] 
**method** | **str** | The algorithm method used for the 3d-reconstruction. | [optional] 
**nominal_pixel_size** | **float** | The nominal pixel size of the projection set of images in Angstroms. | [optional] 
**num_class_averages** | **int** | The number of classes used in the final 3d reconstruction | [optional] 
**num_particles** | **int** | The number of 2D projections or 3D subtomograms used in the 3d reconstruction | [optional] 
**refinement_type** | **str** | Indicates details on how the half-map used for resolution determination (usually by FSC) have been generated. | [optional] 
**resolution** | **float** | The final resolution (in angstroms) of the 3D reconstruction. | [optional] 
**resolution_method** | **str** | The  method used to determine the final resolution  of the 3d reconstruction.  The Fourier Shell Correlation criterion as a measure of  resolution is based on the concept of splitting the (2D)  data set into two halves; averaging each and comparing them  using the Fourier Ring Correlation (FRC) technique. | [optional] 
**symmetry_type** | **str** | The type of symmetry applied to the reconstruction | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

