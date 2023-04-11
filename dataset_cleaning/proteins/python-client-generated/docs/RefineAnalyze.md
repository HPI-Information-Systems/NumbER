# RefineAnalyze

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**luzzati_coordinate_error_free** | **float** | The estimated coordinate error obtained from the plot of  the R value versus sin(theta)/lambda for the reflections  treated as a test set during refinement.   Ref:  Luzzati, V. (1952). Traitement statistique des erreurs  dans la determination des structures cristallines. Acta  Cryst. 5, 802-810. | [optional] 
**luzzati_coordinate_error_obs** | **float** | The estimated coordinate error obtained from the plot of  the R value versus sin(theta)/lambda for reflections classified  as observed.   Ref:  Luzzati, V. (1952). Traitement statistique des erreurs  dans la determination des structures cristallines. Acta  Cryst. 5, 802-810. | [optional] 
**luzzati_d_res_low_free** | **float** | The value of the low-resolution cutoff used in constructing the  Luzzati plot for reflections treated as a test set during  refinement.   Ref:  Luzzati, V. (1952). Traitement statistique des erreurs  dans la determination des structures cristallines. Acta  Cryst. 5, 802-810. | [optional] 
**luzzati_d_res_low_obs** | **float** | The value of the low-resolution cutoff used in  constructing the Luzzati plot for reflections classified as  observed.   Ref:  Luzzati, V. (1952). Traitement statistique des erreurs  dans la determination des structures cristallines. Acta  Cryst. 5, 802-810. | [optional] 
**luzzati_sigma_a_free** | **float** | The value of sigma~a~ used in constructing the Luzzati plot for  the reflections treated as a test set during refinement.  Details of the estimation of sigma~a~ can be specified  in _refine_analyze.Luzzati_sigma_a_free_details.   Ref:  Luzzati, V. (1952). Traitement statistique des erreurs  dans la determination des structures cristallines. Acta  Cryst. 5, 802-810. | [optional] 
**luzzati_sigma_a_obs** | **float** | The value of sigma~a~ used in constructing the Luzzati plot for  reflections classified as observed. Details of the  estimation of sigma~a~ can be specified in  _refine_analyze.Luzzati_sigma_a_obs_details.   Ref:  Luzzati, V. (1952). Traitement statistique des erreurs  dans la determination des structures cristallines. Acta  Cryst. 5, 802-810. | [optional] 
**number_disordered_residues** | **float** | The number of discretely disordered residues in the refined  model. | [optional] 
**occupancy_sum_hydrogen** | **float** | The sum of the occupancies of the hydrogen atoms in the refined  model. | [optional] 
**occupancy_sum_non_hydrogen** | **float** | The sum of the occupancies of the non-hydrogen atoms in the   refined model. | [optional] 
**pdbx_luzzati_d_res_high_obs** | **float** | record the high resolution for calculating Luzzati statistics. | [optional] 
**pdbx_refine_id** | **str** | This data item uniquely identifies a refinement within an entry.  _refine_analyze.pdbx_refine_id can be used to distinguish the results  of joint refinements. | 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

