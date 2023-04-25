# RcsbPolymerEntityInstanceContainerIdentifiers

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**asym_id** | **str** | Instance identifier for this container. | 
**auth_asym_id** | **str** | Author instance identifier for this container. | [optional] 
**auth_to_entity_poly_seq_mapping** | **list[str]** | Residue index mappings between author provided and entity polymer sequence positions.   Author residue indices (auth_seq_num) include insertion codes when present.  The array indices correspond to the indices (1-based) of the deposited sample  sequence (entity_poly_seq). Unmodelled residues are represented with a \&quot;?\&quot; value. | [optional] 
**entity_id** | **str** | Entity identifier for the container. | [optional] 
**entry_id** | **str** | Entry identifier for the container. | 
**rcsb_id** | **str** | A unique identifier for each object in this entity instance container formed by  an &#x27;dot&#x27; (.) separated concatenation of entry and entity instance identifiers. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

