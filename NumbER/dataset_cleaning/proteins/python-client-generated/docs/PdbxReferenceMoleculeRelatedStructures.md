# PdbxReferenceMoleculeRelatedStructures

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**citation_id** | **str** | A link to related reference information in the citation category. | [optional] 
**db_accession** | **str** | The database accession code for the related structure reference. | [optional] 
**db_code** | **str** | The database identifier code for the related structure reference. | [optional] 
**db_name** | **str** | The database name for the related structure reference. | [optional] 
**family_prd_id** | **str** | The value of _pdbx_reference_molecule_related_structures.family_prd_id is a reference to  _pdbx_reference_molecule_list.family_prd_id in category PDBX_REFERENCE_MOLECULE_FAMILY_LIST. | 
**formula** | **str** | The formula for the reference entity. Formulae are written  according to the rules:   1. Only recognised element symbols may be used.   2. Each element symbol is followed by a &#x27;count&#x27; number. A count     of &#x27;1&#x27; may be omitted.   3. A space or parenthesis must separate each element symbol and     its count, but in general parentheses are not used.   4. The order of elements depends on whether or not carbon is     present. If carbon is present, the order should be: C, then     H, then the other elements in alphabetical order of their     symbol. If carbon is not present, the elements are listed     purely in alphabetic order of their symbol. This is the     &#x27;Hill&#x27; system used by Chemical Abstracts. | [optional] 
**name** | **str** | The chemical name for the structure entry in the related database | [optional] 
**ordinal** | **int** | The value of _pdbx_reference_molecule_related_structures.ordinal distinguishes  related structural data for each entity. | 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

