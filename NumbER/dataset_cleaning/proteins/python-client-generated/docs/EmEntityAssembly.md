# EmEntityAssembly

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**details** | **str** | Additional details about the sample or sample subcomponent. | [optional] 
**entity_id_list** | **list[str]** | macromolecules associated with this component, if defined  as comma separated list of entity ids (integers). | [optional] 
**id** | **str** | PRIMARY KEY | 
**name** | **str** | The name of the sample or sample subcomponent. | [optional] 
**oligomeric_details** | **str** | oligomeric details | [optional] 
**parent_id** | **int** | The parent of this assembly.  This data item is an internal category pointer to _em_entity_assembly.id.  By convention, the full assembly (top of hierarchy) is assigned parent id 0 (zero). | [optional] 
**source** | **str** | The type of source (e.g., natural source) for the component (sample or sample subcomponent) | [optional] 
**synonym** | **str** | Alternative name of the component. | [optional] 
**type** | **str** | The general type of the sample or sample subcomponent. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

