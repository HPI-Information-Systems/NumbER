# Symmetry

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**int_tables_number** | **int** | Space-group number from International Tables for Crystallography  Vol. A (2002). | [optional] 
**cell_setting** | **str** | The cell settings for this space-group symmetry. | [optional] 
**pdbx_full_space_group_name_h_m** | **str** | Used for PDB space group:   Example: &#x27;C 1 2 1&#x27;  (instead of C 2)           &#x27;P 1 2 1&#x27;  (instead of P 2)           &#x27;P 1 21 1&#x27; (instead of P 21)           &#x27;P 1 1 21&#x27; (instead of P 21 -unique C axis)           &#x27;H 3&#x27;      (instead of R 3   -hexagonal)           &#x27;H 3 2&#x27;    (instead of R 3 2 -hexagonal) | [optional] 
**space_group_name_h_m** | **str** | Hermann-Mauguin space-group symbol. Note that the  Hermann-Mauguin symbol does not necessarily contain complete  information about the symmetry and the space-group origin. If  used, always supply the FULL symbol from International Tables  for Crystallography Vol. A (2002) and indicate the origin and  the setting if it is not implicit. If there is any doubt that  the equivalent positions can be uniquely deduced from this  symbol, specify the  _symmetry_equiv.pos_as_xyz or  _symmetry.space_group_name_Hall  data items as well. Leave  spaces between symbols referring to  different axes. | [optional] 
**space_group_name_hall** | **str** | Space-group symbol as described by Hall (1981). This symbol  gives the space-group setting explicitly. Leave spaces between  the separate components of the symbol.   Ref: Hall, S. R. (1981). Acta Cryst. A37, 517-525; erratum  (1981) A37, 921. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

