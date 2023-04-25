# PdbxStructAssemblyGen

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**assembly_id** | **str** | This data item is a pointer to _pdbx_struct_assembly.id in the  PDBX_STRUCT_ASSEMBLY category. | [optional] 
**asym_id_list** | **list[str]** | This data item is a pointer to _struct_asym.id in  the STRUCT_ASYM category.   This item may be expressed as a comma separated list of identifiers. | [optional] 
**oper_expression** | **str** | Identifies the operation of collection of operations  from category PDBX_STRUCT_OPER_LIST.   Operation expressions may have the forms:    (1)        the single operation 1   (1,2,5)    the operations 1, 2, 5   (1-4)      the operations 1,2,3 and 4   (1,2)(3,4) the combinations of operations              3 and 4 followed by 1 and 2 (i.e.              the cartesian product of parenthetical              groups applied from right to left) | [optional] 
**ordinal** | **int** | This data item is an ordinal index for the  PDBX_STRUCT_ASSEMBLY category. | 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

