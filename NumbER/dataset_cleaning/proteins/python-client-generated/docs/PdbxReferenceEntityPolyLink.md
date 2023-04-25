# PdbxReferenceEntityPolyLink

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**atom_id_1** | **str** | The atom identifier/name in the first of the two components making  the linkage. | [optional] 
**atom_id_2** | **str** | The atom identifier/name in the second of the two components making  the linkage. | [optional] 
**comp_id_1** | **str** | The component identifier in the first of the two components making the  linkage.   This data item is a pointer to _pdbx_reference_entity_poly_seq.mon_id  in the PDBX_REFERENCE_ENTITY_POLY_SEQ category. | [optional] 
**comp_id_2** | **str** | The component identifier in the second of the two components making the  linkage.   This data item is a pointer to _pdbx_reference_entity_poly_seq.mon_id  in the PDBX_REFERENCE_ENTITY_POLY_SEQ category. | [optional] 
**component_id** | **int** | The entity component identifier entity containing the linkage. | 
**entity_seq_num_1** | **int** | For a polymer entity, the sequence number in the first of  the two components making the linkage.   This data item is a pointer to _pdbx_reference_entity_poly_seq.num  in the PDBX_REFERENCE_ENTITY_POLY_SEQ category. | [optional] 
**entity_seq_num_2** | **int** | For a polymer entity, the sequence number in the second of  the two components making the linkage.   This data item is a pointer to _pdbx_reference_entity_poly_seq.num  in the PDBX_REFERENCE_ENTITY_POLY_SEQ category. | [optional] 
**link_id** | **int** | The value of _pdbx_reference_entity_poly_link.link_id uniquely identifies  a linkage within a polymer entity. | 
**prd_id** | **str** | The value of _pdbx_reference_entity_poly_link.prd_id is a reference  _pdbx_reference_entity_list.prd_id in the PDBX_REFERENCE_ENTITY_POLY category. | 
**ref_entity_id** | **str** | The reference entity id of the polymer entity containing the linkage.   This data item is a pointer to _pdbx_reference_entity_poly.ref_entity_id  in the PDBX_REFERENCE_ENTITY_POLY category. | 
**value_order** | **str** | The bond order target for the non-standard linkage. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

