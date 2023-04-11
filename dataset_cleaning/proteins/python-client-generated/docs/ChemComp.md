# ChemComp

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**formula** | **str** | The formula for the chemical component. Formulae are written  according to the following rules:   (1) Only recognized element symbols may be used.   (2) Each element symbol is followed by a &#x27;count&#x27; number. A count     of &#x27;1&#x27; may be omitted.   (3) A space or parenthesis must separate each cluster of     (element symbol + count), but in general parentheses are     not used.   (4) The order of elements depends on whether carbon is     present or not. If carbon is present, the order should be:     C, then H, then the other elements in alphabetical order     of their symbol. If carbon is not present, the elements     are listed purely in alphabetic order of their symbol. This     is the &#x27;Hill&#x27; system used by Chemical Abstracts. | [optional] 
**formula_weight** | **float** | Formula mass of the chemical component. | [optional] 
**id** | **str** | The value of _chem_comp.id must uniquely identify each item in  the CHEM_COMP list.   For protein polymer entities, this is the three-letter code for  the amino acid.   For nucleic acid polymer entities, this is the one-letter code  for the base. | 
**mon_nstd_parent_comp_id** | **list[str]** | The identifier for the parent component of the nonstandard  component. May be be a comma separated list if this component  is derived from multiple components.   Items in this indirectly point to _chem_comp.id in  the CHEM_COMP category. | [optional] 
**name** | **str** | The full name of the component. | [optional] 
**one_letter_code** | **str** | For standard polymer components, the one-letter code for  the component.   For non-standard polymer components, the  one-letter code for parent component if this exists;  otherwise, the one-letter code should be given as &#x27;X&#x27;.   Components that derived from multiple parents components  are described by a sequence of one-letter-codes. | [optional] 
**pdbx_ambiguous_flag** | **str** | A preliminary classification used by PDB to indicate  that the chemistry of this component while described  as clearly as possible is still ambiguous.  Software  tools may not be able to process this component  definition. | [optional] 
**pdbx_formal_charge** | **int** | The net integer charge assigned to this component. This is the  formal charge assignment normally found in chemical diagrams. | [optional] 
**pdbx_initial_date** | **datetime** | Date component was added to database. | [optional] 
**pdbx_modified_date** | **datetime** | Date component was last modified. | [optional] 
**pdbx_processing_site** | **str** | This data item identifies the deposition site that processed  this chemical component defintion. | [optional] 
**pdbx_release_status** | **str** | This data item holds the current release status for the component. | [optional] 
**pdbx_replaced_by** | **str** | Identifies the _chem_comp.id of the component that  has replaced this component. | [optional] 
**pdbx_replaces** | **str** | Identifies the _chem_comp.id&#x27;s of the components  which have been replaced by this component.  Multiple id codes should be separated by commas. | [optional] 
**pdbx_subcomponent_list** | **str** | The list of subcomponents contained in this component. | [optional] 
**three_letter_code** | **str** | For standard polymer components, the common three-letter code for  the component.   Non-standard polymer components and non-polymer  components are also assigned three-letter-codes.   For ambiguous polymer components three-letter code should  be given as &#x27;UNK&#x27;.  Ambiguous ions are assigned the code &#x27;UNX&#x27;.  Ambiguous non-polymer components are assigned the code &#x27;UNL&#x27;. | [optional] 
**type** | **str** | For standard polymer components, the type of the monomer.  Note that monomers that will form polymers are of three types:  linking monomers, monomers with some type of N-terminal (or 5&#x27;)  cap and monomers with some type of C-terminal (or 3&#x27;) cap. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

