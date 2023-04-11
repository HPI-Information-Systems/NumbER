# RcsbPolymerEntity

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**details** | **str** | A description of special aspects of the entity. | [optional] 
**formula_weight** | **float** | Formula mass (KDa) of the entity. | [optional] 
**pdbx_description** | **str** | A description of the polymer entity. | [optional] 
**pdbx_ec** | **str** | Enzyme Commission (EC) number(s) | [optional] 
**pdbx_fragment** | **str** | Polymer entity fragment description(s). | [optional] 
**pdbx_mutation** | **str** | Details about any polymer entity mutation(s). | [optional] 
**pdbx_number_of_molecules** | **int** | The number of molecules of the entity in the entry. | [optional] 
**rcsb_multiple_source_flag** | **str** | A code indicating the entity has multiple biological sources. | [optional] 
**rcsb_source_part_count** | **int** | The number of biological sources for the polymer entity. Multiple source contributions  may come from the same organism (taxonomy). | [optional] 
**rcsb_source_taxonomy_count** | **int** | The number of distinct source taxonomies for the polymer entity. Commonly used to identify chimeric polymers. | [optional] 
**src_method** | **str** | The method by which the sample for the polymer entity was produced.  Entities isolated directly from natural sources (tissues, soil  samples etc.) are expected to have further information in the  ENTITY_SRC_NAT category. Entities isolated from genetically  manipulated sources are expected to have further information in  the ENTITY_SRC_GEN category. | [optional] 
**rcsb_ec_lineage** | [**list[RcsbPolymerEntityRcsbEcLineage]**](RcsbPolymerEntityRcsbEcLineage.md) |  | [optional] 
**rcsb_macromolecular_names_combined** | [**list[RcsbPolymerEntityRcsbMacromolecularNamesCombined]**](RcsbPolymerEntityRcsbMacromolecularNamesCombined.md) |  | [optional] 
**rcsb_enzyme_class_combined** | [**list[RcsbPolymerEntityRcsbEnzymeClassCombined]**](RcsbPolymerEntityRcsbEnzymeClassCombined.md) |  | [optional] 
**rcsb_polymer_name_combined** | [**RcsbPolymerEntityRcsbPolymerNameCombined**](RcsbPolymerEntityRcsbPolymerNameCombined.md) |  | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

