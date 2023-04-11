# EmImaging

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**accelerating_voltage** | **int** | A value of accelerating voltage (in kV) used for imaging. | [optional] 
**alignment_procedure** | **str** | The type of procedure used to align the microscope electron beam. | [optional] 
**astigmatism** | **str** | astigmatism | [optional] 
**c2_aperture_diameter** | **float** | The open diameter of the c2 condenser lens,  in microns. | [optional] 
**calibrated_defocus_max** | **float** | The maximum calibrated defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus. | [optional] 
**calibrated_defocus_min** | **float** | The minimum calibrated defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus. | [optional] 
**calibrated_magnification** | **int** | The magnification value obtained for a known standard just  prior to, during or just after the imaging experiment. | [optional] 
**cryogen** | **str** | Cryogen type used to maintain the specimen stage temperature during imaging  in the microscope. | [optional] 
**_date** | **datetime** | Date (YYYY-MM-DD) of imaging experiment or the date at which  a series of experiments began. | [optional] 
**details** | **str** | Any additional imaging details. | [optional] 
**detector_distance** | **float** | The camera length (in millimeters). The camera length is the  product of the objective focal length and the combined magnification  of the intermediate and projector lenses when the microscope is  operated in the diffraction mode. | [optional] 
**electron_beam_tilt_params** | **str** | electron beam tilt params | [optional] 
**electron_source** | **str** | The source of electrons. The electron gun. | [optional] 
**id** | **str** | PRIMARY KEY | 
**illumination_mode** | **str** | The mode of illumination. | [optional] 
**microscope_model** | **str** | The name of the model of microscope. | [optional] 
**mode** | **str** | The mode of imaging. | [optional] 
**nominal_cs** | **float** | The spherical aberration coefficient (Cs) in millimeters,  of the objective lens. | [optional] 
**nominal_defocus_max** | **float** | The maximum defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus. | [optional] 
**nominal_defocus_min** | **float** | The minimum defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus. | [optional] 
**nominal_magnification** | **int** | The magnification indicated by the microscope readout. | [optional] 
**recording_temperature_maximum** | **float** | The specimen temperature maximum (kelvin) for the duration  of imaging. | [optional] 
**recording_temperature_minimum** | **float** | The specimen temperature minimum (kelvin) for the duration  of imaging. | [optional] 
**residual_tilt** | **float** | Residual tilt of the electron beam (in miliradians) | [optional] 
**specimen_holder_model** | **str** | The name of the model of specimen holder used during imaging. | [optional] 
**specimen_holder_type** | **str** | The type of specimen holder used during imaging. | [optional] 
**specimen_id** | **str** | Foreign key to the EM_SPECIMEN category | [optional] 
**temperature** | **float** | The mean specimen stage temperature (in kelvin) during imaging  in the microscope. | [optional] 
**tilt_angle_max** | **float** | The maximum angle at which the specimen was tilted to obtain  recorded images. | [optional] 
**tilt_angle_min** | **float** | The minimum angle at which the specimen was tilted to obtain  recorded images. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

