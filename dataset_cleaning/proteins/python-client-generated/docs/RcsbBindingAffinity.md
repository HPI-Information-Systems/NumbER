# RcsbBindingAffinity

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**comp_id** | **str** | Ligand identifier. | 
**type** | **str** | Binding affinity measurement given in one of the following types:  The concentration constants: IC50: the concentration of ligand that reduces enzyme activity by 50%;  EC50: the concentration of compound that generates a half-maximal response;  The binding constant:  Kd: dissociation constant;  Ka: association constant;  Ki: enzyme inhibition constant;  The thermodynamic parameters:  delta G: Gibbs free energy of binding (for association reaction);  delta H: change in enthalpy associated with a chemical reaction;  delta S: change in entropy associated with a chemical reaction. | 
**value** | **float** | Binding affinity value between a ligand and its target molecule. | 
**unit** | **str** | Binding affinity unit.  Dissociation constant Kd is normally in molar units (or millimolar , micromolar, nanomolar, etc).  Association constant Ka is normally expressed in inverse molar units (e.g. M-1). | 
**symbol** | **str** | Binding affinity symbol indicating approximate or precise strength of the binding. | [optional] 
**reference_sequence_identity** | **int** | Data point provided by BindingDB. Percent identity between PDB sequence and reference sequence. | [optional] 
**provenance_code** | **str** | The resource name for the related binding affinity reference. | 
**link** | **str** | Link to external resource referencing the data. | 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

