# PdbxReferenceMolecule

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**chem_comp_id** | **str** | For entities represented as single molecules, the identifier  corresponding to the chemical definition for the molecule. | [optional] 
**_class** | **str** | Broadly defines the function of the entity. | [optional] 
**class_evidence_code** | **str** | Evidence for the assignment of _pdbx_reference_molecule.class | [optional] 
**compound_details** | **str** | Special details about this molecule. | [optional] 
**description** | **str** | Description of this molecule. | [optional] 
**formula** | **str** | The formula for the reference entity. Formulae are written  according to the rules:   1. Only recognised element symbols may be used.   2. Each element symbol is followed by a &#x27;count&#x27; number. A count     of &#x27;1&#x27; may be omitted.   3. A space or parenthesis must separate each element symbol and     its count, but in general parentheses are not used.   4. The order of elements depends on whether or not carbon is     present. If carbon is present, the order should be: C, then     H, then the other elements in alphabetical order of their     symbol. If carbon is not present, the elements are listed     purely in alphabetic order of their symbol. This is the     &#x27;Hill&#x27; system used by Chemical Abstracts. | [optional] 
**formula_weight** | **float** | Formula mass in daltons of the entity. | [optional] 
**name** | **str** | A name of the entity. | [optional] 
**prd_id** | **str** | The value of _pdbx_reference_molecule.prd_id is the unique identifier  for the reference molecule in this family.   By convention this ID uniquely identifies the reference molecule in  in the PDB reference dictionary.   The ID has the template form PRD_dddddd (e.g. PRD_000001) | 
**release_status** | **str** | Defines the current PDB release status for this molecule definition. | [optional] 
**replaced_by** | **str** | Assigns the identifier of the reference molecule that has replaced this molecule. | [optional] 
**replaces** | **str** | Assigns the identifier for the reference molecule which have been replaced  by this reference molecule.  Multiple molecule identifier codes should be separated by commas. | [optional] 
**represent_as** | **str** | Defines how this entity is represented in PDB data files. | [optional] 
**representative_pdb_id_code** | **str** | The PDB accession code for the entry containing a representative example of this molecule. | [optional] 
**type** | **str** | Defines the structural classification of the entity. | [optional] 
**type_evidence_code** | **str** | Evidence for the assignment of _pdbx_reference_molecule.type | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

